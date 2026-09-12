"""Streamlit UI for the Scenario Builder and Fleet Editor.

All actual logic (parsing, validation, persistence) lives in
`quantumfleet.scenarios.*` and `quantumfleet.optimization.problem` -- this
module only renders widgets and wires them to session state / the
repositories, so the logic stays reusable and unit-testable outside of
Streamlit.
"""

import pandas as pd
import streamlit as st

from quantumfleet.fuels.properties import PROPULSION_FUELS
from quantumfleet.optimization.problem import parse_scenario
from quantumfleet.reporting.ports import MAJOR_PORTS, resolve_port_coordinates
from quantumfleet.scenarios.repository import ScenarioRepository, VesselCatalogRepository
from quantumfleet.scenarios.sea_distance import sea_distance_nm
from quantumfleet.scenarios.validation import (
    has_errors,
    validate_scenario_dict,
    validate_scenario_feasibility,
    validate_vessel_dict,
    validate_vessel_speed_coverage,
)

DEFAULT_ROUTE = {
    "route_id": "",
    "distance_nm": 1000.0,
    "cargo_demand_tonnes": 10000.0,
    "period_days": 30.0,
    "max_transit_days": 10.0,
    "allowed_fuel_types": ["HFO", "MDO"],
    "shore_power_available": False,
    "emission_cap_tonnes_co2e": 5000.0,
    "origin_port": "",
    "destination_port": "",
}

DEFAULT_VESSEL = {
    "id": "",
    "type": "Container Ship",
    "tier": "Medium",
    "dwt_tonnes": 40000.0,
    "admiralty_coefficient": 450.0,
    "day_rate_usd": 10000.0,
    "min_speed_knots": 12.0,
    "max_speed_knots": 20.0,
    "aux_load_kw": 700.0,
    "compatible_fuels": ["HFO", "MDO"],
}

VESSEL_TYPES = ["Container Ship", "Bulk Carrier", "Tanker", "RoRo", "General Cargo"]
VESSEL_TIERS = ["Small", "Medium", "Large"]


def _blank_scenario() -> dict:
    return {"name": "", "description": "", "carbon_price_usd_per_tonne": 50.0, "green_fuel_pathway": False, "routes": []}


def _normalize_scenario_dict(raw: dict) -> dict:
    """Fills in every field the builder's widgets expect, from a raw dict
    that might be missing optional keys (e.g. a scenario written before a
    field like `description` existed)."""
    routes = []
    for r in raw.get("routes") or []:
        merged = dict(DEFAULT_ROUTE)
        merged.update(r)
        merged["allowed_fuel_types"] = list(r.get("allowed_fuel_types") or [])
        routes.append(merged)
    return {
        "name": raw.get("name", ""),
        "description": raw.get("description", ""),
        "carbon_price_usd_per_tonne": raw.get("carbon_price_usd_per_tonne", 50.0),
        "green_fuel_pathway": raw.get("green_fuel_pathway", False),
        "routes": routes,
    }


def _init_builder_state() -> None:
    if "builder_scenario" not in st.session_state:
        st.session_state["builder_scenario"] = _blank_scenario()
        st.session_state["builder_key"] = None  # None = unsaved/new draft


def _render_issues(issues: list, header: str) -> None:
    if not issues:
        return
    st.write(f"**{header}**")
    for issue in issues:
        (st.error if issue.severity == "error" else st.warning)(f"`{issue.field}` -- {issue.message}")


def render_scenario_builder(scenario_repo: ScenarioRepository, vessel_classes: dict, speed_bins: list[float]) -> None:
    """Renders the full Scenario Builder tab. On a successful save, sets
    `st.session_state["active_scenario_key"]` itself and reruns -- the
    sidebar's scenario list is computed earlier in script execution order,
    so without an explicit rerun a just-saved scenario would not appear
    there until some unrelated later interaction happened to trigger one."""
    _init_builder_state()
    draft = st.session_state["builder_scenario"]

    if "scenario_saved_message" in st.session_state:
        st.success(st.session_state.pop("scenario_saved_message"))

    st.subheader("Load, start new, duplicate, or delete")
    all_scenarios = scenario_repo.list_all()
    labels: dict[str, str | None] = {"-- New blank scenario --": None}
    for s in all_scenarios:
        tag = "built-in" if s.is_builtin else "yours"
        labels[f"{s.name}  ({tag}, {s.n_routes} routes)"] = s.key
    choice = st.selectbox("Choose a scenario", list(labels.keys()), key="builder_load_choice")
    chosen_key = labels[choice]

    lc1, lc2, lc3 = st.columns(3)
    if lc1.button("Load into builder"):
        if chosen_key is None:
            st.session_state["builder_scenario"] = _blank_scenario()
            st.session_state["builder_key"] = None
        else:
            st.session_state["builder_scenario"] = _normalize_scenario_dict(scenario_repo.load_raw(chosen_key))
            st.session_state["builder_key"] = chosen_key
        st.rerun()

    if lc2.button("Duplicate as new scenario", disabled=chosen_key is None):
        source_name = scenario_repo.load_raw(chosen_key).get("name", "scenario")
        new_key = scenario_repo.duplicate(chosen_key, f"{source_name} (copy)")
        st.session_state["builder_scenario"] = _normalize_scenario_dict(scenario_repo.load_raw(new_key))
        st.session_state["builder_key"] = new_key
        st.success(f"Duplicated as '{source_name} (copy)'.")
        st.rerun()

    if lc3.button("Delete this scenario", disabled=chosen_key is None):
        if chosen_key is not None and not scenario_repo.is_builtin(chosen_key):
            scenario_repo.delete_user_scenario(chosen_key)
            if st.session_state.get("builder_key") == chosen_key:
                st.session_state["builder_scenario"] = _blank_scenario()
                st.session_state["builder_key"] = None
            st.success("Deleted.")
            st.rerun()
        else:
            st.warning("Built-in scenarios can't be deleted -- duplicate it first if you want to modify it.")

    st.divider()
    st.subheader("Scenario details")
    draft["name"] = st.text_input("Scenario name", value=draft.get("name", ""), key="builder_name")
    draft["description"] = st.text_area("Description (optional)", value=draft.get("description", ""), height=68, key="builder_description")
    dc1, dc2 = st.columns(2)
    draft["carbon_price_usd_per_tonne"] = dc1.number_input("Carbon price (USD/t CO2e)", min_value=0.0, value=float(draft.get("carbon_price_usd_per_tonne", 50.0)), step=10.0, key="builder_carbon_price")
    draft["green_fuel_pathway"] = dc2.checkbox("Use green (low-carbon) production pathway for alt fuels", value=bool(draft.get("green_fuel_pathway", False)), key="builder_green_pathway")

    st.divider()
    st.subheader(f"Routes ({len(draft['routes'])})")
    _render_routes_editor(draft)

    st.divider()
    issues = validate_scenario_dict(draft)
    feasibility_issues = []
    if not has_errors(issues):
        try:
            scenario_config = parse_scenario(draft)
            feasibility_issues = validate_scenario_feasibility(scenario_config, vessel_classes, speed_bins)
        except (KeyError, ValueError, TypeError) as e:
            issues = list(issues)
            from quantumfleet.scenarios.validation import ValidationIssue

            issues.append(ValidationIssue("routes", f"Could not process this scenario: {e}"))

    _render_issues(issues, "Fix these before saving:")
    _render_issues(feasibility_issues, "Feasibility warnings (optimization may find no plan, even though the scenario is well-formed):")
    if not issues and not feasibility_issues:
        st.success("No validation issues.")

    can_save = not has_errors(issues)
    if st.button("Save scenario", type="primary", disabled=not can_save):
        key = st.session_state["builder_key"] or scenario_repo.unique_key(draft["name"])
        try:
            scenario_repo.save_user_scenario(key, draft)
            st.session_state["builder_key"] = key
            st.session_state["active_scenario_key"] = key
            st.session_state["scenario_saved_message"] = f"Saved '{draft['name']}'. It's now selected in the sidebar, exactly like a built-in scenario."
            st.rerun()
        except ValueError as e:
            st.error(str(e))


_PORT_EXAMPLES = "Shanghai, Singapore, Rotterdam, Mumbai, Chennai, Hong Kong, Dubai, New York, Los Angeles"

_PORT_HELP = (
    "Type a major port name -- a port, not a country. When both ports are recognized, "
    "the distance below is filled in automatically from the real sea route (around land "
    "and through the appropriate canal or strait), and the route appears on the map in "
    f"Scenario & routes. Recognized names include: {_PORT_EXAMPLES}. An unrecognized name "
    "is still accepted; you just set the distance yourself and the route won't be mapped."
)


def _sync_sea_distance(route: dict, i: int) -> float | None:
    """Auto-fill `distance_nm` when the user switches to a recognized port pair.

    Returns the sea distance for the current pair, or None if either port is
    unknown. Must be called BEFORE the distance number_input is created: it
    writes into that widget's session-state slot, which Streamlit reads when
    instantiating the widget.
    """
    origin = route.get("origin_port") or ""
    destination = route.get("destination_port") or ""
    pair = (origin.strip().lower(), destination.strip().lower())
    pair_key = f"route_{i}_portpair"
    computed = sea_distance_nm(origin, destination)

    # The "use this distance" button fires after the distance widget already
    # exists, and Streamlit forbids writing a widget's session-state slot once
    # instantiated. So the button only records an intent and reruns; the write
    # itself happens here, before the widget is built.
    pending = st.session_state.pop(f"route_{i}_pending_dist", None)
    if pending is not None:
        st.session_state[f"route_{i}_dist"] = pending
        route["distance_nm"] = pending

    if pair_key not in st.session_state:
        # First render of this route -- typically loaded from a saved scenario.
        # Adopt whatever distance it already has instead of overwriting a
        # deliberately stored value on mere page load.
        st.session_state[pair_key] = pair
    elif st.session_state[pair_key] != pair:
        st.session_state[pair_key] = pair
        if computed is not None:
            st.session_state[f"route_{i}_dist"] = round(computed, 1)
            route["distance_nm"] = round(computed, 1)
    return computed


def _render_sea_distance_note(route: dict, i: int, computed: float | None) -> None:
    """Explain where the distance came from, or why it could not be filled in."""
    origin = (route.get("origin_port") or "").strip()
    destination = (route.get("destination_port") or "").strip()
    if not origin or not destination:
        return

    unknown = [name for name in (origin, destination) if resolve_port_coordinates(name) is None]
    if unknown:
        names = " and ".join(f"'{name}'" for name in unknown)
        st.caption(f":material/help: {names} is not a known port, so the distance stays manual. Recognized names include: {_PORT_EXAMPLES}.")
        return

    current = float(route.get("distance_nm") or 0.0)
    if abs(current - computed) <= 1.0:
        st.caption(f":material/route: Sea route {origin} → {destination}: **{computed:,.0f} nm**, filled in automatically.")
        return

    # The field was edited by hand after the auto-fill. Leave it alone, but
    # offer a one-click way back to the computed value.
    note_col, button_col = st.columns([3, 1])
    note_col.caption(f":material/route: The sea route {origin} → {destination} is **{computed:,.0f} nm**, but this field says {current:,.0f} nm.")
    if button_col.button(f"Use {computed:,.0f} nm", key=f"route_{i}_usedist"):
        st.session_state[f"route_{i}_pending_dist"] = round(computed, 1)
        st.rerun()


def _render_routes_editor(draft: dict) -> None:
    routes = draft["routes"]
    to_delete = None
    to_duplicate = None

    for i, route in enumerate(routes):
        label = f"Route {i + 1}: {route.get('route_id') or '(unnamed)'}"
        with st.expander(label, expanded=False):
            route["route_id"] = st.text_input("Route ID (unique)", value=route.get("route_id", ""), key=f"route_{i}_id")

            pc1, pc2 = st.columns(2)
            route["origin_port"] = pc1.text_input("Origin port (optional)", value=route.get("origin_port") or "", key=f"route_{i}_origin", help=_PORT_HELP)
            route["destination_port"] = pc2.text_input("Destination port (optional)", value=route.get("destination_port") or "", key=f"route_{i}_dest", help=_PORT_HELP)

            sea_distance = _sync_sea_distance(route, i)

            nc1, nc2 = st.columns(2)
            route["distance_nm"] = nc1.number_input("Distance (nautical miles)", min_value=0.0, value=float(route.get("distance_nm", 0.0)), key=f"route_{i}_dist")
            route["cargo_demand_tonnes"] = nc2.number_input("Cargo demand (tonnes per period)", min_value=0.0, value=float(route.get("cargo_demand_tonnes", 0.0)), key=f"route_{i}_demand")
            _render_sea_distance_note(route, i, sea_distance)

            nc3, nc4 = st.columns(2)
            route["period_days"] = nc3.number_input("Planning period (days)", min_value=0.0, value=float(route.get("period_days", 30.0)), key=f"route_{i}_period")
            route["max_transit_days"] = nc4.number_input("Max transit time (days)", min_value=0.0, value=float(route.get("max_transit_days", 10.0)), key=f"route_{i}_transit")

            route["emission_cap_tonnes_co2e"] = st.number_input("Emission cap (tonnes CO2e per period)", min_value=0.0, value=float(route.get("emission_cap_tonnes_co2e", 1000.0)), key=f"route_{i}_cap")
            route["allowed_fuel_types"] = st.multiselect("Allowed fuel types", options=list(PROPULSION_FUELS), default=route.get("allowed_fuel_types", []), key=f"route_{i}_fuels")
            route["shore_power_available"] = st.checkbox("Shore power available at berth", value=bool(route.get("shore_power_available", False)), key=f"route_{i}_shore")

            bc1, bc2 = st.columns(2)
            if bc1.button("Duplicate this route", key=f"route_{i}_dup"):
                to_duplicate = i
            if bc2.button("Delete this route", key=f"route_{i}_del"):
                to_delete = i

    if to_delete is not None:
        routes.pop(to_delete)
        st.rerun()
    if to_duplicate is not None:
        copy = dict(routes[to_duplicate])
        copy["route_id"] = f"{copy['route_id']}_copy"
        routes.insert(to_duplicate + 1, copy)
        st.rerun()

    if st.button("+ Add route"):
        new_route = dict(DEFAULT_ROUTE)
        new_route["route_id"] = f"ROUTE_{len(routes) + 1}"
        routes.append(new_route)
        st.rerun()


def _render_vessel_fields(v: dict, key_prefix: str) -> None:
    v["id"] = st.text_input("Vessel ID (unique)", value=v.get("id", ""), key=f"{key_prefix}_id")
    c1, c2 = st.columns(2)
    v["type"] = c1.selectbox("Type", VESSEL_TYPES, index=VESSEL_TYPES.index(v["type"]) if v.get("type") in VESSEL_TYPES else 0, key=f"{key_prefix}_type")
    v["tier"] = c2.selectbox("Tier", VESSEL_TIERS, index=VESSEL_TIERS.index(v["tier"]) if v.get("tier") in VESSEL_TIERS else 1, key=f"{key_prefix}_tier")

    c3, c4 = st.columns(2)
    v["dwt_tonnes"] = c3.number_input("Capacity / DWT (tonnes)", min_value=0.0, value=float(v.get("dwt_tonnes", 40000.0)), key=f"{key_prefix}_dwt")
    v["day_rate_usd"] = c4.number_input("Charter cost (USD/day)", min_value=0.0, value=float(v.get("day_rate_usd", 10000.0)), key=f"{key_prefix}_rate")

    c5, c6 = st.columns(2)
    v["min_speed_knots"] = c5.number_input("Min speed (knots)", min_value=0.0, value=float(v.get("min_speed_knots", 10.0)), key=f"{key_prefix}_minspeed")
    v["max_speed_knots"] = c6.number_input("Max speed (knots)", min_value=0.0, value=float(v.get("max_speed_knots", 20.0)), key=f"{key_prefix}_maxspeed")

    c7, c8 = st.columns(2)
    v["admiralty_coefficient"] = c7.number_input("Admiralty coefficient (fuel-efficiency factor -- built-ins use ~380/450/520 for Small/Medium/Large)", min_value=1.0, value=float(v.get("admiralty_coefficient", 450.0)), key=f"{key_prefix}_adm")
    v["aux_load_kw"] = c8.number_input("Auxiliary / hotel load (kW, used at berth)", min_value=0.0, value=float(v.get("aux_load_kw", 700.0)), key=f"{key_prefix}_aux")

    v["compatible_fuels"] = st.multiselect("Compatible fuels", options=list(PROPULSION_FUELS), default=v.get("compatible_fuels", []), key=f"{key_prefix}_fuels")


def render_fleet_editor(vessel_repo: VesselCatalogRepository, speed_bins: list[float]) -> None:
    st.subheader("Built-in vessel classes (read-only)")
    builtin, _ = vessel_repo.load_builtin()
    st.dataframe(pd.DataFrame([v.__dict__ for v in builtin.values()]), width="stretch")
    st.caption("Note: engine power isn't a stored field -- it's derived from capacity, speed, and the admiralty coefficient by the physics model, the same way for every vessel.")

    st.divider()
    st.subheader("Your custom vessel classes")
    user_entries = vessel_repo.load_user_raw()
    if not user_entries:
        st.caption("None yet -- add one below.")

    to_delete = None
    for i, v in enumerate(user_entries):
        with st.expander(f"{v.get('id') or '(unnamed)'}", expanded=False):
            _render_vessel_fields(v, key_prefix=f"uv_{i}")
            bc1, bc2, bc3 = st.columns(3)
            if bc1.button("Save changes", key=f"uv_{i}_save"):
                issues = validate_vessel_dict(v) + validate_vessel_speed_coverage(v, speed_bins)
                if has_errors(issues):
                    for issue in issues:
                        st.error(f"{issue.field}: {issue.message}")
                else:
                    vessel_repo.save_user_vessel(v)
                    st.success("Saved.")
                    st.rerun()
            if bc2.button("Duplicate", key=f"uv_{i}_dup"):
                copy = dict(v)
                copy["id"] = f"{copy['id']}_copy"
                vessel_repo.save_user_vessel(copy)
                st.rerun()
            if bc3.button("Delete", key=f"uv_{i}_del"):
                to_delete = v["id"]

    if to_delete is not None:
        vessel_repo.delete_user_vessel(to_delete)
        st.rerun()

    st.divider()
    st.subheader("Add a new vessel class")
    nonce = st.session_state.get("new_vessel_nonce", 0)
    new_vessel = dict(DEFAULT_VESSEL)
    _render_vessel_fields(new_vessel, key_prefix=f"newv_{nonce}")
    if st.button("Add vessel", type="primary"):
        existing_ids = set(vessel_repo.load_merged()[0].keys())
        issues = validate_vessel_dict(new_vessel, existing_ids=existing_ids) + validate_vessel_speed_coverage(new_vessel, speed_bins)
        if has_errors(issues):
            for issue in issues:
                st.error(f"{issue.field}: {issue.message}")
        else:
            vessel_repo.save_user_vessel(new_vessel)
            st.session_state["new_vessel_nonce"] = nonce + 1  # forces the form back to blank defaults
            st.success(f"Added '{new_vessel['id']}'.")
            st.rerun()
