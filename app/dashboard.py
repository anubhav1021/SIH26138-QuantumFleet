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

st.set_page_config(page_title="Quantum-Inspired Green Fleet Optimizer", layout="wide")

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


st.title("Quantum-Inspired Fuel Consumption Prediction & Green Fleet Optimization")
st.caption("SIH26138 -- decision-support platform for fleet deployment, alternative fuels, and emissions scenario analysis.")

all_scenario_summaries = scenario_repo.list_all()
scenario_option_labels = {f"{s.name}  ({'built-in' if s.is_builtin else 'yours'}, {s.n_routes} routes)": s.key for s in all_scenario_summaries}
scenario_keys_in_order = list(scenario_option_labels.values())

with st.sidebar:
    st.header("Scenario")
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

    run_clicked = st.button("Run optimization", type="primary")

problem = _build_active_problem(scenario_key, carbon_price, green_pathway)

_feasibility_warnings = validate_scenario_feasibility(problem.scenario, problem.vessel_classes, list(problem.speed_bins_knots))
if _feasibility_warnings:
    st.sidebar.warning(f"{len(_feasibility_warnings)} feasibility warning(s) for this scenario -- see the Scenario Builder tab for detail.")

if run_clicked:
    with st.spinner(f"Running quantum-inspired optimizer on {len(problem.scenario.routes)} routes..."):
        optimizer = QuantumEvolutionaryOptimizer(problem=problem, population_size=population_size, n_generations=n_generations, seed=int(seed))
        archive, history = optimizer.run()
    st.session_state["archive"] = archive
    st.session_state["history"] = history
    st.session_state["problem"] = problem

tab_builder, tab_fleet, tab_results, tab_whatif, tab_scenario, tab_prediction, tab_benchmark, tab_about = st.tabs(
    ["Scenario Builder", "Fleet Editor", "Optimization Results", "What-If Analysis", "Scenario & Routes", "Prediction Model", "Benchmarking", "Methodology"]
)

with tab_builder:
    st.write("Create, edit, duplicate, or delete scenarios -- no YAML editing required. Saved scenarios appear in the sidebar selector immediately, exactly like the built-in ones.")
    builder_ui.render_scenario_builder(scenario_repo, problem.vessel_classes, list(problem.speed_bins_knots))

with tab_fleet:
    st.write("Add, edit, duplicate, or delete vessel classes. Built-in vessel classes can't be modified or removed, but your custom ones are used by the optimizer exactly the same way.")
    builder_ui.render_fleet_editor(vessel_repo, list(problem.speed_bins_knots))

with tab_results:
    if "archive" not in st.session_state:
        st.info("Configure a scenario in the sidebar and click **Run optimization** to see results.")
    else:
        archive = st.session_state["archive"]
        history = st.session_state["history"]
        run_problem = st.session_state["problem"]
        feasible = [e for e in archive.entries if e.objectives.violation <= 1e-9]

        c1, c2, c3 = st.columns(3)
        c1.metric("Non-dominated plans", len(archive))
        c2.metric("Fully feasible", len(feasible))
        c3.metric("Final hypervolume", f"{history[-1].hypervolume:,.0f}")

        if len(archive) == 1:
            st.info("Only one non-dominated solution survived. That usually means the constraints and objectives left little room for tradeoffs -- check the Scenario Builder tab for feasibility warnings, or widen the schedule/emission cap to open up more options.")

        st.plotly_chart(pareto_front_scatter_3d(archive), width="stretch")
        st.plotly_chart(convergence_line_chart({"Quantum-Inspired (QEA)": history}), width="stretch")

        st.subheader("Compare Pareto extremes")
        st.caption("The single archive members that minimize each objective on its own -- a quick way to see the range of the tradeoff before picking a priority below.")
        extremes = pick_extreme_solutions(archive)
        ext_cols = st.columns(3)
        for col, label, key in [(ext_cols[0], "Lowest fuel", "lowest_fuel"), (ext_cols[1], "Lowest cost", "lowest_cost"), (ext_cols[2], "Lowest emissions", "lowest_emissions")]:
            entry = extremes[key]
            if entry is not None:
                col.metric(label, f"{entry.objectives.fuel_tonnes:,.0f} t fuel", f"${entry.objectives.cost_usd:,.0f} / {entry.objectives.co2e_tonnes:,.0f} t CO2e")

        st.subheader("Optimization priority")
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
        st.subheader("Recommended fleet plan")
        if o.violation > 1e-9:
            st.warning(f"This is the LEAST INFEASIBLE plan found (violation score {o.violation:.3f}), not a feasible one.")
            st.write("**Why it's infeasible:**")
            for reason in explain_infeasibility(summary["route_feasibilities"]):
                st.write(f"- {reason}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Fuel", f"{o.fuel_tonnes:,.1f} t")
        m2.metric("Lifecycle CO2e", f"{o.co2e_tonnes:,.1f} t")
        m3.metric("Total cost", f"${o.cost_usd:,.0f}")
        m4.metric("of which carbon cost", f"${summary['carbon_cost_usd']:,.0f}")

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Cargo fulfillment", f"{summary['cargo_fulfillment_pct']:.0f}%")
        m6.metric("Schedule OK", f"{summary['schedule_compliant_routes']}/{summary['n_routes']} routes")
        m7.metric("Within emission cap", f"{summary['emission_compliant_routes']}/{summary['n_routes']} routes")
        m8.metric("Vessels deployed", summary["vessels_used"])

        if summary["fuel_mix"]:
            st.caption("Fuel mix (vessel count): " + ", ".join(f"{fuel} x{count}" for fuel, count in summary["fuel_mix"].items()))

        plan_df = per_assignment_breakdown(selected.plan, run_problem)
        st.dataframe(plan_df, width="stretch")
        if not plan_df.empty:
            st.plotly_chart(emissions_by_fuel_bar(plan_df), width="stretch")

        st.subheader("Why this plan")
        for line in explain_why_plan_chosen(o, summary["route_feasibilities"], archive, selected):
            st.write(line)

        st.divider()
        st.subheader("Baseline vs. optimized")
        st.caption("Baseline = a greedy, single-vessel-type-per-route heuristic (conventional route planning, no metaheuristic search) -- see docs/algorithm_details.md.")
        if st.button("Compute baseline comparison"):
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
        if st.button("Export report"):
            report_dir = PROJECT_ROOT / "results" / "dashboard_export"
            report_path = build_report(run_problem, archive, history, str(report_dir))
            st.success(f"Report written to {report_path}")
            st.download_button("Download report.md", data=Path(report_path).read_text(encoding="utf-8"), file_name="report.md")

with tab_whatif:
    analysis_ui.render_what_if(problem, population_size, n_generations, int(seed))

with tab_scenario:
    st.subheader(f"Routes in '{problem.scenario.name}'")

    route_map_fig = route_map(list(problem.scenario.routes))
    if route_map_fig is not None:
        st.plotly_chart(route_map_fig, width="stretch")
    else:
        st.caption("No route in this scenario has a recognized origin/destination port name (or explicit coordinates) set, so no map is shown -- add them in the Scenario Builder to enable this.")

    route_ids = [r.route_id for r in problem.scenario.routes]
    selected_route_id = st.selectbox("Inspect a route", route_ids, key="scenario_tab_route_select")
    selected_route = next(r for r in problem.scenario.routes if r.route_id == selected_route_id)
    rc1, rc2, rc3, rc4 = st.columns(4)
    rc1.metric("Origin", selected_route.origin_port or "(not set)")
    rc2.metric("Destination", selected_route.destination_port or "(not set)")
    rc3.metric("Distance", f"{selected_route.distance_nm:,.0f} nm")
    rc4.metric("Cargo demand", f"{selected_route.cargo_demand_tonnes:,.0f} t")
    rc5, rc6, rc7 = st.columns(3)
    rc5.metric("Max transit", f"{selected_route.max_transit_days:.1f} days")
    rc6.metric("Emission cap", f"{selected_route.emission_cap_tonnes_co2e:,.0f} t CO2e")
    rc7.metric("Shore power", "Available" if selected_route.shore_power_available else "Not available")
    st.caption(f"Allowed fuels: {', '.join(selected_route.allowed_fuel_types)}")

    st.subheader("All routes")
    routes_df = pd.DataFrame([r.__dict__ for r in problem.scenario.routes])
    st.dataframe(routes_df, width="stretch")
    st.subheader("Vessel class catalog")
    vessels_df = pd.DataFrame([v.__dict__ for v in problem.vessel_classes.values()])
    st.dataframe(vessels_df, width="stretch")

with tab_prediction:
    st.write("Generate a synthetic voyage dataset and train the fuel-consumption prediction models.")
    n_vessels = st.slider("Synthetic vessels", 20, 300, 150, key="pred_n_vessels")
    legs = st.slider("Legs per vessel", 5, 40, 20, key="pred_legs")

    if st.button("Generate data & train models"):
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
        st.subheader("Residual analysis: is there a learnable physics-model bias?")
        r2 = detail["residual_r2"]
        if r2 < 0.1:
            st.info(f"Cross-validated R² of predicting (actual - physics) from features: **{r2:.3f}** -- near zero or negative, meaning the residual is essentially unpredictable noise here. The hybrid model performs on par with physics alone on this synthetic data; see docs/algorithm_details.md for why, and why this is expected to differ on real voyage data.")
        else:
            st.success(f"Cross-validated R² of predicting (actual - physics) from features: **{r2:.3f}** -- a meaningfully learnable bias exists, and the hybrid model should outperform physics alone.")

        c1, c2 = st.columns(2)
        c1.plotly_chart(actual_vs_predicted_scatter(detail["y_test"], detail["rf_pred"], title="Random Forest: actual vs. predicted"), width="stretch")
        c2.plotly_chart(residual_scatter(detail["y_test"], detail["rf_pred"]), width="stretch")

        if hasattr(detail["rf_model"], "feature_importances_"):
            st.plotly_chart(feature_importance_bar(detail["rf_model"].feature_importances_, detail["feature_names"]), width="stretch")

with tab_benchmark:
    st.write("Compare the quantum-inspired optimizer against classical baselines on the same scenario (a small, fast configuration for interactive use).")
    bench_routes = st.slider("Routes for this comparison", 2, 20, 5, key="bench_routes")
    bench_pop = st.slider("Population", 10, 60, 20, key="bench_pop")
    bench_gen = st.slider("Generations", 10, 100, 30, key="bench_gen")

    if st.button("Run benchmark comparison"):
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
            rows.append({"algorithm": name, "archive_size": len(arc), "feasible": feasible_n, "final_hypervolume": hist[-1].hypervolume})
        rows.append({"algorithm": "Greedy Heuristic", "archive_size": 1, "feasible": int(greedy_obj.violation <= 1e-9), "final_hypervolume": np.nan})

        st.dataframe(pd.DataFrame(rows), width="stretch")
        st.plotly_chart(
            convergence_line_chart({"Quantum-Inspired (QEA)": qea_hist, "Classical GA": ga_hist, "Random Search": rand_hist}),
            width="stretch",
        )

with tab_about:
    st.markdown(
        """
### Methodology

- **Fuel prediction**: an Admiralty-formula physics baseline plus a Random-Forest / Gradient-Boosting
  data-driven model, with hyperparameters tuned by Quantum-behaved Particle Swarm Optimization (QPSO).
- **Optimization formulation**: decision variables are vessel mix, capacity, cruising speed, fuel type,
  and shore-power use per route; objectives are fuel, lifecycle CO2e, and cost; constraints are cargo
  demand, schedule reliability, and emission caps.
- **Quantum-inspired engine**: a qubit-encoded evolutionary algorithm (rotation-gate updates,
  quantum-NOT mutation) driving an external Pareto archive, with a composite-leader recombination step
  and a catastrophe operator for population diversity.
- **Benchmarking**: the same encoding, constraints, and archive power a classical genetic algorithm,
  random search, and a greedy single-vessel-type heuristic, so comparisons isolate what the
  quantum-inspired mechanism specifically contributes.

See `docs/algorithm_details.md` and `docs/implementation_guide.md` for full detail.
        """
    )
