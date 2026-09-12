"""Validation for scenarios and vessels before they reach the optimizer.

Two layers: structural (does this raw dict have the right shape/types at
all?) and semantic/feasibility (do the values make sense together, and could
a plan conceivably exist?). Messages are written for a route planner, not a
Python developer -- nobody should need to read a traceback to understand
what's wrong with a scenario they built in the UI.
"""

from dataclasses import dataclass

from quantumfleet.fuels.properties import PROPULSION_FUELS
from quantumfleet.optimization import constraints as con
from quantumfleet.optimization.encoding import Assignment
from quantumfleet.optimization.problem import N_SLOTS_PER_ROUTE, Route, ScenarioConfig
from quantumfleet.prediction.physics_model import VesselClass

REQUIRED_ROUTE_FIELDS = [
    "route_id", "distance_nm", "cargo_demand_tonnes", "period_days",
    "max_transit_days", "allowed_fuel_types", "shore_power_available", "emission_cap_tonnes_co2e",
]
REQUIRED_VESSEL_FIELDS = [
    "id", "type", "tier", "dwt_tonnes", "admiralty_coefficient",
    "day_rate_usd", "min_speed_knots", "max_speed_knots", "aux_load_kw", "compatible_fuels",
]


@dataclass(frozen=True)
class ValidationIssue:
    field: str
    message: str
    severity: str = "error"  # "error" blocks optimization; "warning" doesn't

    @property
    def is_error(self) -> bool:
        return self.severity == "error"


def has_errors(issues: list[ValidationIssue]) -> bool:
    return any(i.is_error for i in issues)


def validate_route_dict(r: dict, existing_route_ids: set[str] | None = None) -> list[ValidationIssue]:
    issues = [ValidationIssue(f, f"'{f}' is required.") for f in REQUIRED_ROUTE_FIELDS if r.get(f) in (None, "")]
    if issues:
        return issues  # can't safely check values below without the fields present

    route_id = str(r["route_id"])
    if existing_route_ids and route_id in existing_route_ids:
        issues.append(ValidationIssue("route_id", f"Route ID '{route_id}' is already used by another route in this scenario -- route IDs must be unique."))

    if float(r["distance_nm"]) <= 0:
        issues.append(ValidationIssue("distance_nm", "Distance must be a positive number of nautical miles."))
    if float(r["cargo_demand_tonnes"]) < 0:
        issues.append(ValidationIssue("cargo_demand_tonnes", "Cargo demand cannot be negative."))
    if float(r["period_days"]) <= 0:
        issues.append(ValidationIssue("period_days", "Planning period must be a positive number of days."))
    if float(r["max_transit_days"]) <= 0:
        issues.append(ValidationIssue("max_transit_days", "Maximum transit time must be a positive number of days."))
    if float(r["emission_cap_tonnes_co2e"]) <= 0:
        issues.append(ValidationIssue("emission_cap_tonnes_co2e", "Emission cap should be a positive number of tonnes CO2e.", severity="warning"))

    if not r["allowed_fuel_types"]:
        issues.append(ValidationIssue("allowed_fuel_types", "Select at least one allowed fuel type for this route."))
    else:
        unknown = [f for f in r["allowed_fuel_types"] if f not in PROPULSION_FUELS]
        if unknown:
            issues.append(ValidationIssue("allowed_fuel_types", f"Unknown fuel type(s): {', '.join(unknown)}."))

    return issues


def validate_vessel_dict(v: dict, existing_ids: set[str] | None = None) -> list[ValidationIssue]:
    issues = [ValidationIssue(f, f"'{f}' is required.") for f in REQUIRED_VESSEL_FIELDS if v.get(f) in (None, "")]
    if issues:
        return issues

    vessel_id = str(v["id"])
    if existing_ids and vessel_id in existing_ids:
        issues.append(ValidationIssue("id", f"Vessel ID '{vessel_id}' is already in use -- vessel IDs must be unique."))
    if float(v["dwt_tonnes"]) <= 0:
        issues.append(ValidationIssue("dwt_tonnes", "Capacity (DWT) must be a positive number of tonnes."))
    if float(v["admiralty_coefficient"]) <= 0:
        issues.append(ValidationIssue("admiralty_coefficient", "Admiralty coefficient must be positive."))
    if float(v["day_rate_usd"]) < 0:
        issues.append(ValidationIssue("day_rate_usd", "Charter cost cannot be negative."))
    if float(v["min_speed_knots"]) <= 0 or float(v["max_speed_knots"]) <= 0:
        issues.append(ValidationIssue("min_speed_knots", "Speeds must be positive."))
    elif float(v["min_speed_knots"]) >= float(v["max_speed_knots"]):
        issues.append(ValidationIssue("min_speed_knots", "Minimum speed must be less than maximum speed."))
    if float(v["aux_load_kw"]) < 0:
        issues.append(ValidationIssue("aux_load_kw", "Auxiliary (hotel) load cannot be negative."))

    if not v["compatible_fuels"]:
        issues.append(ValidationIssue("compatible_fuels", "Select at least one compatible fuel type."))
    else:
        unknown = [f for f in v["compatible_fuels"] if f not in PROPULSION_FUELS]
        if unknown:
            issues.append(ValidationIssue("compatible_fuels", f"Unknown fuel type(s): {', '.join(unknown)}."))

    return issues


def validate_vessel_speed_coverage(v: dict, speed_bins_knots: list[float]) -> list[ValidationIssue]:
    """A vessel whose [min_speed, max_speed] range contains none of the
    platform's global speed options can never be assigned a valid speed by
    the optimizer's repair step -- catch this at save time, not silently."""
    if v.get("min_speed_knots") is None or v.get("max_speed_knots") is None:
        return []
    covers_any = any(float(v["min_speed_knots"]) <= s <= float(v["max_speed_knots"]) for s in speed_bins_knots)
    if not covers_any:
        return [ValidationIssue("min_speed_knots", f"This vessel's speed range doesn't include any of the platform's speed options ({', '.join(str(s) for s in speed_bins_knots)} knots) -- it would never be assignable. Widen the range to include at least one.")]
    return []


def validate_scenario_dict(raw: dict) -> list[ValidationIssue]:
    issues = []
    if not (raw.get("name") or "").strip():
        issues.append(ValidationIssue("name", "Scenario needs a name."))

    routes = raw.get("routes") or []
    if not routes:
        issues.append(ValidationIssue("routes", "Add at least one route."))

    seen_ids: set[str] = set()
    for i, r in enumerate(routes):
        for issue in validate_route_dict(r, existing_route_ids=seen_ids):
            issues.append(ValidationIssue(f"routes[{i}].{issue.field}", issue.message, issue.severity))
        if r.get("route_id"):
            seen_ids.add(str(r["route_id"]))

    if float(raw.get("carbon_price_usd_per_tonne", 0) or 0) < 0:
        issues.append(ValidationIssue("carbon_price_usd_per_tonne", "Carbon price cannot be negative."))

    return issues


def max_feasible_capacity_tonnes(route: Route, vessel_classes: dict[str, VesselClass], speed_bins: list[float]) -> float:
    """A generous UPPER BOUND on how much cargo this route could possibly
    deliver per period: the single best compatible vessel, at its fastest
    schedule-compliant speed, at maximum count, replicated across every
    assignment slot. If actual demand exceeds even this, no optimizer run
    could ever satisfy it.

    This is necessary-but-not-sufficient: passing the check doesn't
    guarantee the optimizer will FIND a feasible plan (that still depends on
    search), only that one could theoretically exist. It's surfaced as a
    pre-check warning, not a hard gate, for exactly that reason.
    """
    compatible = [v for v in vessel_classes.values() if set(v.compatible_fuels) & set(route.allowed_fuel_types)]
    if not compatible:
        return 0.0

    best = 0.0
    for vessel in compatible:
        for speed in speed_bins:
            if not (vessel.min_speed_knots <= speed <= vessel.max_speed_knots):
                continue
            if con.one_way_transit_days(route.distance_nm, speed) > route.max_transit_days:
                continue
            probe = Assignment(route_id=route.route_id, vessel_class=vessel.id, fuel_type=vessel.compatible_fuels[0], speed_knots=speed, use_shore_power=route.shore_power_available, count=7)
            trips = con.trips_per_period(route, probe)
            capacity = vessel.dwt_tonnes * probe.count * trips * N_SLOTS_PER_ROUTE
            best = max(best, capacity)
    return best


def validate_scenario_feasibility(scenario: ScenarioConfig, vessel_classes: dict[str, VesselClass], speed_bins: list[float]) -> list[ValidationIssue]:
    """Semantic/feasibility checks once the scenario parses successfully."""
    issues = []
    for route in scenario.routes:
        compatible_any = any(set(v.compatible_fuels) & set(route.allowed_fuel_types) for v in vessel_classes.values())
        if not compatible_any:
            issues.append(
                ValidationIssue(
                    f"routes.{route.route_id}.allowed_fuel_types",
                    f"No vessel in the fleet supports any of this route's allowed fuels ({', '.join(route.allowed_fuel_types)}) -- no plan could ever be built for this route.",
                )
            )
            continue

        ceiling = max_feasible_capacity_tonnes(route, vessel_classes, speed_bins)
        if route.cargo_demand_tonnes > ceiling:
            issues.append(
                ValidationIssue(
                    f"routes.{route.route_id}.cargo_demand_tonnes",
                    f"Cargo demand ({route.cargo_demand_tonnes:,.0f} t) exceeds the maximum this route's "
                    f"compatible fleet could plausibly deliver per period (~{ceiling:,.0f} t), even using the "
                    f"largest compatible vessel at maximum count across all slots. Increase the schedule/period, "
                    f"add a bigger or faster compatible vessel, or lower the demand.",
                    severity="warning",
                )
            )
    return issues
