"""Streamlit UI for What-If scenario comparison: apply a delta to the active
scenario's carbon price, alt-fuel production pathway, and a uniform
cargo-demand / emission-cap scaling across all routes, then run the SAME
optimizer on both the original and modified scenario and show a real,
computed comparison -- never a fabricated one.

Kept as thin UI glue: the actual comparison math lives in
`quantumfleet.reporting.explain` (percentage-change logic identical to the
baseline-vs-optimized comparison) and the modified scenario is built with
`dataclasses.replace` over the same `ScenarioConfig`/`Route` the rest of the
platform uses -- no parallel scenario representation.
"""

from dataclasses import replace

import pandas as pd
import streamlit as st

from quantumfleet.optimization.problem import ProblemSpec, build_problem
from quantumfleet.optimization.qea import QuantumEvolutionaryOptimizer
from quantumfleet.reporting.explain import explain_infeasibility, explain_plan_feasibility


def _best_entry(archive):
    feasible = [e for e in archive.entries if e.objectives.violation <= 1e-9]
    pool = feasible if feasible else archive.entries
    return min(pool, key=lambda e: e.objectives.cost_usd), len(feasible)


def _pct_change(base_value: float, new_value: float) -> float | None:
    if base_value == 0:
        return None
    return 100.0 * (new_value - base_value) / base_value


def render_what_if(base_problem: ProblemSpec, population_size: int, n_generations: int, seed: int) -> None:
    st.write(
        "Change one or more assumptions below, then run the optimizer on both the current scenario and the "
        "modified ('what-if') version to see a real, computed comparison -- e.g. \"what happens if carbon "
        "price rises 50%?\""
    )

    c1, c2 = st.columns(2)
    carbon_price_delta_pct = c1.slider("Carbon price change (%)", -100, 300, 50, step=10, key="whatif_carbon_pct")
    pathway_choice = c2.selectbox("Alt-fuel production pathway", ["Keep current", "Force grey (conventional)", "Force green (low-carbon)"], key="whatif_pathway")

    c3, c4 = st.columns(2)
    demand_delta_pct = c3.slider("Cargo demand change, all routes (%)", -50, 100, 0, step=5, key="whatif_demand_pct")
    cap_delta_pct = c4.slider("Emission cap change, all routes (%)", -50, 100, 0, step=5, key="whatif_cap_pct")

    if st.button("Run what-if comparison", type="primary"):
        new_carbon_price = base_problem.scenario.carbon_price_usd_per_tonne * (1 + carbon_price_delta_pct / 100)
        if pathway_choice.startswith("Force grey"):
            new_green_pathway = False
        elif pathway_choice.startswith("Force green"):
            new_green_pathway = True
        else:
            new_green_pathway = base_problem.scenario.green_fuel_pathway

        new_routes = tuple(
            replace(
                r,
                cargo_demand_tonnes=max(0.0, r.cargo_demand_tonnes * (1 + demand_delta_pct / 100)),
                emission_cap_tonnes_co2e=max(0.0, r.emission_cap_tonnes_co2e * (1 + cap_delta_pct / 100)),
            )
            for r in base_problem.scenario.routes
        )
        new_scenario = replace(base_problem.scenario, carbon_price_usd_per_tonne=new_carbon_price, green_fuel_pathway=new_green_pathway, routes=new_routes)
        whatif_problem = build_problem(new_scenario, base_problem.vessel_classes, list(base_problem.speed_bins_knots))

        with st.spinner(f"Running the optimizer on both scenarios ({len(base_problem.scenario.routes)} routes each)..."):
            base_archive, _ = QuantumEvolutionaryOptimizer(problem=base_problem, population_size=population_size, n_generations=n_generations, seed=seed).run()
            whatif_archive, _ = QuantumEvolutionaryOptimizer(problem=whatif_problem, population_size=population_size, n_generations=n_generations, seed=seed).run()

        st.session_state["whatif_result"] = {"base_archive": base_archive, "whatif_archive": whatif_archive, "whatif_problem": whatif_problem}

    if "whatif_result" in st.session_state:
        _render_comparison(st.session_state["whatif_result"])


def _render_comparison(result: dict) -> None:
    base_best, base_feasible_n = _best_entry(result["base_archive"])
    whatif_best, whatif_feasible_n = _best_entry(result["whatif_archive"])

    rows = []
    for label, key in [("Fuel (t)", "fuel_tonnes"), ("CO2e (t)", "co2e_tonnes"), ("Cost (USD)", "cost_usd")]:
        base_v = getattr(base_best.objectives, key)
        whatif_v = getattr(whatif_best.objectives, key)
        pct = _pct_change(base_v, whatif_v)
        rows.append(
            {
                "Metric": label,
                "Current scenario": f"{base_v:,.1f}",
                "What-if scenario": f"{whatif_v:,.1f}",
                "Change": f"{whatif_v - base_v:+,.1f}",
                "% change": f"{pct:+.1f}%" if pct is not None else "N/A (zero baseline)",
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    fc1, fc2 = st.columns(2)
    fc1.metric("Feasible plans -- current scenario", base_feasible_n)
    fc2.metric("Feasible plans -- what-if scenario", whatif_feasible_n)

    if whatif_feasible_n == 0:
        st.warning("The what-if scenario has NO feasible plan under these settings. Why:")
        feasibilities = explain_plan_feasibility(whatif_best.plan, result["whatif_problem"])
        for reason in explain_infeasibility(feasibilities):
            st.write(f"- {reason}")
