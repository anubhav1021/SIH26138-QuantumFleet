"""Streamlit decision-support dashboard for the quantum-inspired green fleet
optimizer. Single page with tabs (kept deliberately to one page rather than
Streamlit's multi-page mechanism -- avoids cross-page session-state friction
for what's fundamentally one workflow: configure a scenario, run the
optimizer, inspect the result).

Run with: streamlit run app/dashboard.py
"""

import sys
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(APP_DIR))

import numpy as np
import pandas as pd
import streamlit as st

import analysis_ui
import scenario_builder_ui as builder_ui
from quantumfleet.benchmarking.baselines_optimization import ClassicalGeneticAlgorithm, run_greedy_heuristic, run_random_search
from quantumfleet.benchmarking.baselines_prediction import physics_baseline_predictions
from quantumfleet.data_generation.generator import generate_voyage_records
from quantumfleet.data_generation.validate import assert_valid_voyage_records
from quantumfleet.optimization.problem import build_problem
from quantumfleet.optimization.qea import QuantumEvolutionaryOptimizer, evaluate
from quantumfleet.prediction.evaluate import compare_models, regression_metrics
from quantumfleet.prediction.features import build_feature_matrix, build_target, feature_names, make_encoder, train_test_split_by_vessel
from quantumfleet.prediction.hybrid_model import HybridFuelPredictionModel, residual_learnability_r2
from quantumfleet.prediction.ml_model import FuelPredictionModel
from quantumfleet.reporting.charts import (
    actual_vs_predicted_scatter,
    archive_to_dataframe,
    convergence_line_chart,
    emissions_by_fuel_bar,
    feature_importance_bar,
    pareto_front_scatter_3d,
    residual_scatter,
    route_map,
)
from quantumfleet.reporting.explain import compare_baseline_vs_optimized, explain_infeasibility, explain_why_plan_chosen, pick_extreme_solutions, plan_summary_metrics
from quantumfleet.reporting.report_builder import build_report, per_assignment_breakdown
from quantumfleet.scenarios.repository import ScenarioRepository, VesselCatalogRepository
from quantumfleet.scenarios.validation import has_errors, validate_scenario_feasibility

st.set_page_config(
    page_title="Quantum-Inspired Green Fleet Optimizer",
    page_icon=":material/directions_boat:",
    layout="wide",
    initial_sidebar_state="expanded",
)

CONFIGS_DIR = PROJECT_ROOT / "configs"
scenario_repo = ScenarioRepository(builtin_dir=CONFIGS_DIR, user_dir=CONFIGS_DIR / "user_scenarios")
vessel_repo = VesselCatalogRepository(builtin_path=CONFIGS_DIR / "vessel_types.yaml", user_path=CONFIGS_DIR / "user_vessels.yaml")

# Deliberately NOT cached: scenario/vessel YAML files can be edited and
# re-saved under the same key from the Scenario Builder / Fleet Editor tabs
# within the same session. st.cache_data keys purely on argument values, not
# file mtimes, so caching this by key would keep serving a stale scenario
# after an edit -- the parse itself is cheap enough (milliseconds) that
# caching buys nothing worth that correctness risk.
def _build_active_problem(scenario_key: str, carbon_price: float, green_pathway: bool):
    scenario = scenario_repo.load_scenario_config(scenario_key)
    scenario = replace(scenario, carbon_price_usd_per_tonne=carbon_price, green_fuel_pathway=green_pathway)
    vessel_classes, speed_bins = vessel_repo.load_merged()
    return build_problem(scenario, vessel_classes, speed_bins)


def _weighted_best_entry(archive, w_fuel: float, w_co2e: float, w_cost: float):
    df = archive_to_dataframe(archive)
    pool = df[df["feasible"]] if df["feasible"].any() else df
    cols = ["fuel_tonnes", "co2e_tonnes", "cost_usd"]
    span = pool[cols].max() - pool[cols].min()
    norm = (pool[cols] - pool[cols].min()) / span.replace(0, 1.0)
    score = w_fuel * norm["fuel_tonnes"] + w_co2e * norm["co2e_tonnes"] + w_cost * norm["cost_usd"]
    return archive.entries[score.idxmin()]


st.title("Quantum Fleet Optimizer", anchor=False)
st.caption("SIH26138 — decision-support platform for fleet deployment, alternative fuels and emissions analysis")

all_scenario_summaries = scenario_repo.list_all()
scenario_option_labels = {f"{s.name}  ({'built-in' if s.is_builtin else 'yours'}, {s.n_routes} routes)": s.key for s in all_scenario_summaries}
scenario_keys_in_order = list(scenario_option_labels.values())

with st.sidebar:
    st.subheader(":material/settings: Configuration", anchor=False)
    default_key = st.session_state.get("active_scenario_key", scenario_keys_in_order[0])
    default_index = scenario_keys_in_order.index(default_key) if default_key in scenario_keys_in_order else 0
    scenario_label = st.selectbox("Scenario", list(scenario_option_labels.keys()), index=default_index)
    scenario_key = scenario_option_labels[scenario_label]
    st.session_state["active_scenario_key"] = scenario_key

    if st.session_state.get("_results_computed_for_scenario") not in (None, scenario_key):
        # Switching scenarios invalidates any previously computed results --
        # without this, e.g. a "Baseline vs. optimized" table computed for
        # the OLD scenario would keep showing under the newly selected one,
        # silently mislabeled. Optimization results, once computed, keep
        # their own `problem` reference (used throughout the Results tab) so
        # they stay internally consistent even after this clears; the point
        # is to stop carrying them forward as if they were the NEW scenario.
        for key in ("archive", "history", "problem", "baseline_comparison", "whatif_result", "prediction_results", "prediction_detail"):
            st.session_state.pop(key, None)
    st.session_state["_results_computed_for_scenario"] = scenario_key

    carbon_price = st.number_input("Carbon price (USD/t CO2e)", min_value=0.0, max_value=500.0, value=50.0, step=10.0)
    green_pathway = st.checkbox("Use green (low-carbon) production pathway for alt fuels", value=False)

    with st.expander("Advanced optimization settings"):
        st.caption("The quantum-inspired search itself always remains multi-objective -- these only control how long/wide it searches.")
        population_size = st.slider("Population size", 10, 100, 40)
        n_generations = st.slider("Generations", 10, 200, 50)
        seed = st.number_input("Random seed", min_value=0, value=42, step=1)

    run_clicked = st.button("Run optimization", type="primary", icon=":material/rocket_launch:", width="stretch")

problem = _build_active_problem(scenario_key, carbon_price, green_pathway)

_feasibility_warnings = validate_scenario_feasibility(problem.scenario, problem.vessel_classes, list(problem.speed_bins_knots))
if _feasibility_warnings:
    st.sidebar.warning(
        f"{len(_feasibility_warnings)} feasibility warning(s) for this scenario — see the Scenario builder tab for detail.",
        icon=":material/warning:",
    )

if run_clicked:
    with st.spinner(f"Running quantum-inspired optimizer on {len(problem.scenario.routes)} routes..."):
        optimizer = QuantumEvolutionaryOptimizer(problem=problem, population_size=population_size, n_generations=n_generations, seed=int(seed))
        archive, history = optimizer.run()
    st.session_state["archive"] = archive
    st.session_state["history"] = history
    st.session_state["problem"] = problem

tab_builder, tab_fleet, tab_results, tab_whatif, tab_scenario, tab_prediction, tab_benchmark, tab_about = st.tabs([
    ":material/edit_document: Scenario builder",
    ":material/directions_boat: Fleet editor",
    ":material/insights: Optimization results",
    ":material/alt_route: What-if analysis",
    ":material/map: Scenario & routes",
    ":material/model_training: Prediction model",
    ":material/speed: Benchmarking",
    ":material/menu_book: Methodology",
])

with tab_builder:
    st.caption("Create, edit, duplicate, or delete scenarios — no YAML editing required. Saved scenarios appear in the sidebar immediately.")
    builder_ui.render_scenario_builder(scenario_repo, problem.vessel_classes, list(problem.speed_bins_knots))

with tab_fleet:
    st.caption("Add, edit, duplicate, or delete vessel classes. Custom vessels are used by the optimizer exactly like built-in ones.")
    builder_ui.render_fleet_editor(vessel_repo, list(problem.speed_bins_knots))

with tab_results:
    if "archive" not in st.session_state:
        st.info(
            "Configure a scenario in the sidebar and click **Run optimization** to see results.",
            icon=":material/rocket_launch:",
        )
    else:
        archive = st.session_state["archive"]
        history = st.session_state["history"]
        run_problem = st.session_state["problem"]
        feasible = [e for e in archive.entries if e.objectives.violation <= 1e-9]

        c1, c2, c3 = st.columns(3)
        c1.metric("Non-dominated plans", len(archive), border=True, help="Size of the Pareto archive — distinct plans where no objective can improve without another worsening.")
        c2.metric("Fully feasible", len(feasible), border=True, help="Archive members violating no cargo, schedule, or emission-cap constraint.")
        # Hypervolume is a raw volume in (fuel x CO2e x cost) space, so it runs
        # to ~1e15 -- shown in scientific notation because the full integer is
        # 16 digits wide and unreadable in a metric card.
        c3.metric("Final hypervolume", f"{history[-1].hypervolume:.3g}", border=True, help="Volume of objective space dominated by the archive. Higher is better; the absolute scale is only meaningful when comparing runs on the same scenario.")

        if len(archive) == 1:
            st.warning(
                "Only one non-dominated solution survived. Check the Scenario builder tab for feasibility warnings, or widen the schedule/emission cap to open up more options.",
                icon=":material/warning:",
            )

        st.plotly_chart(pareto_front_scatter_3d(archive), width="stretch")
        st.plotly_chart(convergence_line_chart({"Quantum-Inspired (QEA)": history}), width="stretch")

        st.subheader(":material/compare_arrows: Compare Pareto extremes", anchor=False)
        st.caption("The single archive members that minimize each objective independently — a quick way to see the range of tradeoffs before picking a priority.")
        extremes = pick_extreme_solutions(archive)
        ext_cols = st.columns(3)
        for col, label, key in [(ext_cols[0], "Lowest fuel", "lowest_fuel"), (ext_cols[1], "Lowest cost", "lowest_cost"), (ext_cols[2], "Lowest emissions", "lowest_emissions")]:
            entry = extremes[key]
            if entry is not None:
                # Cost/CO2e go in a caption rather than st.metric's `delta`:
                # delta renders a trend arrow, which would read as "increased"
                # on what is really just this plan's other two objectives.
                with col.container(border=True):
                    st.metric(label, f"{entry.objectives.fuel_tonnes:,.0f} t fuel")
                    st.caption(f"${entry.objectives.cost_usd:,.0f}  ·  {entry.objectives.co2e_tonnes:,.0f} t CO2e")

        st.subheader(":material/tune: Optimization priority", anchor=False)
        priority_presets = {
            "Balanced": (1 / 3, 1 / 3, 1 / 3),
            "Lowest fuel": (1.0, 0.0, 0.0),
            "Lowest emissions": (0.0, 1.0, 0.0),
            "Lowest cost": (0.0, 0.0, 1.0),
            "Custom weights": None,
        }
        priority_choice = st.selectbox("Pick one plan from the Pareto front by priority", list(priority_presets.keys()))
        if priority_presets[priority_choice] is not None:
            w_fuel, w_co2e, w_cost = priority_presets[priority_choice]
        else:
            wc1, wc2, wc3 = st.columns(3)
            w_fuel = wc1.slider("Fuel importance", 0.0, 1.0, 0.34)
            w_co2e = wc2.slider("Emissions importance", 0.0, 1.0, 0.33)
            w_cost = wc3.slider("Cost importance", 0.0, 1.0, 0.33)

        selected = _weighted_best_entry(archive, w_fuel, w_co2e, w_cost)
        o = selected.objectives
        summary = plan_summary_metrics(selected.plan, o, run_problem)

        st.divider()
        st.subheader(":material/checklist: Recommended fleet plan", anchor=False)
        if o.violation > 1e-9:
            st.warning(
                f"This is the **least infeasible** plan found (violation score {o.violation:.3f}), not a feasible one.",
                icon=":material/warning:",
            )
            st.write("**Why it's infeasible:**")
            for reason in explain_infeasibility(summary["route_feasibilities"]):
                st.write(f"- {reason}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Fuel", f"{o.fuel_tonnes:,.1f} t", border=True)
        m2.metric("Lifecycle CO2e", f"{o.co2e_tonnes:,.1f} t", border=True)
        m3.metric("Total cost", f"${o.cost_usd:,.0f}", border=True)
        m4.metric("of which carbon cost", f"${summary['carbon_cost_usd']:,.0f}", border=True)

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Cargo fulfillment", f"{summary['cargo_fulfillment_pct']:.0f}%", border=True)
        m6.metric("Schedule OK", f"{summary['schedule_compliant_routes']}/{summary['n_routes']} routes", border=True)
        m7.metric("Within emission cap", f"{summary['emission_compliant_routes']}/{summary['n_routes']} routes", border=True)
        m8.metric("Vessels deployed", summary["vessels_used"], border=True)

        if summary["fuel_mix"]:
            st.caption("Fuel mix (vessel count): " + ", ".join(f"{fuel} x{count}" for fuel, count in summary["fuel_mix"].items()))

        plan_df = per_assignment_breakdown(selected.plan, run_problem)
        st.dataframe(plan_df, width="stretch")
        if not plan_df.empty:
            st.plotly_chart(emissions_by_fuel_bar(plan_df), width="stretch")

        st.subheader(":material/lightbulb: Why this plan", anchor=False)
        for line in explain_why_plan_chosen(o, summary["route_feasibilities"], archive, selected):
            st.write(line)

        st.divider()
        st.subheader(":material/bar_chart: Baseline vs. optimized", anchor=False)
        st.caption("Baseline = a greedy, single-vessel-type-per-route heuristic (conventional route planning). See docs/algorithm_details.md for details.")
        if st.button("Compute baseline comparison", icon=":material/play_arrow:"):
            with st.spinner("Running the greedy baseline heuristic..."):
                st.session_state["baseline_comparison"] = compare_baseline_vs_optimized(run_problem, o)

        if "baseline_comparison" in st.session_state:
            comparison = st.session_state["baseline_comparison"]
            rows = []
            for label, key, pct_key in [("Fuel (t)", "fuel_tonnes", "fuel_improvement_pct"), ("CO2e (t)", "co2e_tonnes", "co2e_improvement_pct"), ("Cost (USD)", "cost_usd", "cost_improvement_pct")]:
                baseline_v = getattr(comparison["baseline"], key)
                optimized_v = getattr(comparison["optimized"], key)
                pct = comparison[pct_key]
                rows.append(
                    {
                        "Metric": label,
                        "Baseline (greedy)": f"{baseline_v:,.1f}",
                        "Quantum-inspired": f"{optimized_v:,.1f}",
                        "Absolute change": f"{optimized_v - baseline_v:+,.1f}",
                        "% improvement": f"{pct:+.1f}%" if pct is not None else "N/A (baseline is zero)",
                    }
                )
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

        st.divider()
        if st.button("Export report", icon=":material/description:"):
            report_dir = PROJECT_ROOT / "results" / "dashboard_export"
            report_path = build_report(run_problem, archive, history, str(report_dir))
            st.success(f"Report written to {report_path}", icon=":material/check_circle:")
            st.download_button("Download report.md", data=Path(report_path).read_text(encoding="utf-8"), file_name="report.md", icon=":material/download:")

with tab_whatif:
    analysis_ui.render_what_if(problem, population_size, n_generations, int(seed))

with tab_scenario:
    st.subheader(f":material/map: Routes in '{problem.scenario.name}'", anchor=False)

    route_map_fig = route_map(list(problem.scenario.routes))
    if route_map_fig is not None:
        st.plotly_chart(route_map_fig, width="stretch")
    else:
        st.caption("No route in this scenario has a recognized origin/destination port name (or explicit coordinates) set, so no map is shown -- add them in the Scenario Builder to enable this.")

    route_ids = [r.route_id for r in problem.scenario.routes]
    selected_route_id = st.selectbox("Inspect a route", route_ids, key="scenario_tab_route_select")
    selected_route = next(r for r in problem.scenario.routes if r.route_id == selected_route_id)
    rc1, rc2, rc3, rc4 = st.columns(4)
    rc1.metric("Origin", selected_route.origin_port or "Not set", border=True)
    rc2.metric("Destination", selected_route.destination_port or "Not set", border=True)
    rc3.metric("Distance", f"{selected_route.distance_nm:,.0f} nm", border=True)
    rc4.metric("Cargo demand", f"{selected_route.cargo_demand_tonnes:,.0f} t", border=True)
    rc5, rc6, rc7 = st.columns(3)
    rc5.metric("Max transit", f"{selected_route.max_transit_days:.1f} days", border=True)
    rc6.metric("Emission cap", f"{selected_route.emission_cap_tonnes_co2e:,.0f} t CO2e", border=True)
    rc7.metric("Shore power", "Available" if selected_route.shore_power_available else "Not available", border=True)
    st.caption(f"Allowed fuels: {', '.join(selected_route.allowed_fuel_types)}")

    st.subheader(":material/table_rows: All routes", anchor=False)
    # Built column-by-column rather than from __dict__ so the lat/lon plumbing
    # used by the map stays out of the table and the tuple of allowed fuels
    # renders as text instead of a Python repr.
    routes_df = pd.DataFrame(
        [
            {
                "Route": r.route_id,
                "Origin": r.origin_port or "—",
                "Destination": r.destination_port or "—",
                "Distance (nm)": r.distance_nm,
                "Cargo (t)": r.cargo_demand_tonnes,
                "Period (d)": r.period_days,
                "Max transit (d)": r.max_transit_days,
                "Emission cap (t CO2e)": r.emission_cap_tonnes_co2e,
                "Shore power": r.shore_power_available,
                "Allowed fuels": ", ".join(r.allowed_fuel_types),
            }
            for r in problem.scenario.routes
        ]
    )
    st.dataframe(
        routes_df,
        width="stretch",
        hide_index=True,
        column_config={
            "Distance (nm)": st.column_config.NumberColumn(format="localized"),
            "Cargo (t)": st.column_config.NumberColumn(format="localized"),
            "Period (d)": st.column_config.NumberColumn(format="%.1f"),
            "Max transit (d)": st.column_config.NumberColumn(format="%.1f"),
            "Emission cap (t CO2e)": st.column_config.NumberColumn(format="localized"),
            "Shore power": st.column_config.CheckboxColumn(),
        },
    )

    st.subheader(":material/directions_boat: Vessel class catalog", anchor=False)
    vessels_df = pd.DataFrame([v.__dict__ for v in problem.vessel_classes.values()])
    st.dataframe(vessels_df, width="stretch", hide_index=True)

with tab_prediction:
    st.caption("Generate a synthetic voyage dataset and train the fuel-consumption prediction models.")
    n_vessels = st.slider("Synthetic vessels", 20, 300, 150, key="pred_n_vessels")
    legs = st.slider("Legs per vessel", 5, 40, 20, key="pred_legs")

    if st.button("Generate data & train models", type="primary", icon=":material/model_training:"):
        with st.spinner("Generating synthetic data and training models..."):
            df = generate_voyage_records(problem.vessel_classes, n_vessels=n_vessels, legs_per_vessel=legs, seed=int(seed))
            assert_valid_voyage_records(df)
            train_df, test_df = train_test_split_by_vessel(df, test_size=0.2, seed=int(seed))
            encoder = make_encoder(list(problem.vessel_classes.keys()))
            X_train, y_train = build_feature_matrix(train_df, encoder), build_target(train_df)
            X_test, y_test = build_feature_matrix(test_df, encoder), build_target(test_df)

            results = {
                "Physics (Admiralty)": regression_metrics(y_test, physics_baseline_predictions(test_df, problem.vessel_classes)),
            }
            fitted_models = {}
            for kind, label in [("linear", "Linear Regression"), ("random_forest", "Random Forest"), ("gradient_boosting", "Gradient Boosting")]:
                model = FuelPredictionModel(kind=kind).fit(X_train, y_train)
                results[label] = regression_metrics(y_test, model.predict(X_test))
                fitted_models[label] = model

            hybrid = HybridFuelPredictionModel().fit(train_df, X_train, problem.vessel_classes)
            hybrid_pred = hybrid.predict(test_df, X_test, problem.vessel_classes)
            results["Hybrid (physics + RF residual)"] = regression_metrics(y_test, hybrid_pred)
            residual_r2 = residual_learnability_r2(train_df, X_train, problem.vessel_classes)

        st.session_state["prediction_results"] = compare_models(results)
        st.session_state["prediction_detail"] = {
            "y_test": y_test,
            "rf_pred": fitted_models["Random Forest"].predict(X_test),
            "rf_model": fitted_models["Random Forest"].estimator,
            "feature_names": feature_names(encoder),
            "residual_r2": residual_r2,
        }

    if "prediction_results" in st.session_state:
        st.dataframe(st.session_state["prediction_results"].style.format("{:.4f}"), width="stretch")

        detail = st.session_state["prediction_detail"]
        st.subheader(":material/scatter_plot: Residual analysis", anchor=False)
        r2 = detail["residual_r2"]
        if r2 < 0.1:
            st.info(
                f"Cross-validated R² of predicting (actual − physics) from features: **{r2:.3f}** — near zero or negative, meaning the residual is essentially unpredictable noise here. The hybrid model performs on par with physics alone on this synthetic data; see docs/algorithm_details.md for why, and why this is expected to differ on real voyage data.",
                icon=":material/info:",
            )
        else:
            st.success(
                f"Cross-validated R² of predicting (actual − physics) from features: **{r2:.3f}** — a meaningfully learnable bias exists, and the hybrid model should outperform physics alone.",
                icon=":material/check_circle:",
            )

        c1, c2 = st.columns(2)
        c1.plotly_chart(actual_vs_predicted_scatter(detail["y_test"], detail["rf_pred"], title="Random Forest: actual vs. predicted"), width="stretch")
        c2.plotly_chart(residual_scatter(detail["y_test"], detail["rf_pred"]), width="stretch")

        if hasattr(detail["rf_model"], "feature_importances_"):
            st.plotly_chart(feature_importance_bar(detail["rf_model"].feature_importances_, detail["feature_names"]), width="stretch")

with tab_benchmark:
    st.caption("Compare the quantum-inspired optimizer against classical baselines on the same scenario (a small, fast configuration for interactive use).")
    bench_routes = st.slider("Routes for this comparison", 2, 20, 5, key="bench_routes")
    bench_pop = st.slider("Population", 10, 60, 20, key="bench_pop")
    bench_gen = st.slider("Generations", 10, 100, 30, key="bench_gen")

    if st.button("Run benchmark comparison", type="primary", icon=":material/speed:"):
        from dataclasses import replace as _replace

        from quantumfleet.benchmarking.runner import scaled_problem

        instance = scaled_problem(problem, bench_routes)
        with st.spinner("Running QEA, classical GA, and random search..."):
            qea_archive, qea_hist = QuantumEvolutionaryOptimizer(problem=instance, population_size=bench_pop, n_generations=bench_gen, seed=int(seed)).run()
            ga_archive, ga_hist = ClassicalGeneticAlgorithm(problem=instance, population_size=bench_pop, n_generations=bench_gen, seed=int(seed)).run()
            rand_archive, rand_hist = run_random_search(instance, population_size=bench_pop, n_generations=bench_gen, seed=int(seed))
            greedy_plan = run_greedy_heuristic(instance)
            greedy_obj = evaluate(greedy_plan, instance)

        rows = []
        for name, arc, hist in [("Quantum-Inspired (QEA)", qea_archive, qea_hist), ("Classical GA", ga_archive, ga_hist), ("Random Search", rand_archive, rand_hist)]:
            feasible_n = sum(1 for e in arc.entries if e.objectives.violation <= 1e-9)
            rows.append({"Algorithm": name, "Archive size": len(arc), "Feasible": feasible_n, "Final hypervolume": hist[-1].hypervolume})
        rows.append({"Algorithm": "Greedy heuristic", "Archive size": 1, "Feasible": int(greedy_obj.violation <= 1e-9), "Final hypervolume": np.nan})

        st.dataframe(
            pd.DataFrame(rows),
            width="stretch",
            hide_index=True,
            column_config={
                "Final hypervolume": st.column_config.NumberColumn(
                    format="localized",
                    help="Higher is better. Undefined for the greedy heuristic, which returns a single solution rather than a front.",
                )
            },
        )
        st.plotly_chart(
            convergence_line_chart({"Quantum-Inspired (QEA)": qea_hist, "Classical GA": ga_hist, "Random Search": rand_hist}),
            width="stretch",
        )

with tab_about:
    st.caption("How each graded deliverable is implemented. See docs/algorithm_details.md and docs/implementation_guide.md for the full write-up.")

    sections = [
        (
            ":material/science:",
            "Fuel prediction",
            "Physics baseline (Admiralty formula) plus ML models (Random Forest / Gradient Boosting), with hyperparameters tuned by Quantum-behaved Particle Swarm Optimization (QPSO).",
        ),
        (
            ":material/functions:",
            "Optimization formulation",
            "Decision variables: vessel mix, capacity, cruising speed, fuel type, and shore-power use per route. Objectives: fuel, lifecycle CO₂e, cost. Constraints: cargo demand, schedule reliability, emission caps.",
        ),
        (
            ":material/hub:",
            "Quantum-inspired engine",
            "Qubit-encoded evolutionary algorithm with rotation-gate updates, quantum-NOT mutation, an external Pareto archive, composite-leader recombination, and a catastrophe operator for diversity.",
        ),
        (
            ":material/speed:",
            "Benchmarking",
            "The same encoding, constraints, and archive drive a classical GA, random search, and a greedy heuristic — a fair comparison that isolates what the quantum-inspired mechanism actually contributes.",
        ),
    ]

    left_col, right_col = st.columns(2)
    for i, (icon, title, content) in enumerate(sections):
        with left_col if i % 2 == 0 else right_col:
            with st.container(border=True):
                st.markdown(f"##### {icon} {title}")
                st.caption(content)
