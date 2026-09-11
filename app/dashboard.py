"""Streamlit decision-support dashboard for the quantum-inspired green fleet
optimizer. Single page with tabs (kept deliberately to one page rather than
Streamlit's multi-page mechanism -- avoids cross-page session-state friction
for what's fundamentally one workflow: configure a scenario, run the
optimizer, inspect the result).

Run with: streamlit run app/dashboard.py
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd
import streamlit as st

from quantumfleet.benchmarking.baselines_optimization import ClassicalGeneticAlgorithm, run_greedy_heuristic, run_random_search
from quantumfleet.data_generation.generator import generate_voyage_records
from quantumfleet.data_generation.validate import assert_valid_voyage_records
from quantumfleet.optimization.problem import ProblemSpec, load_problem
from quantumfleet.optimization.qea import QuantumEvolutionaryOptimizer, evaluate
from quantumfleet.prediction.evaluate import compare_models, regression_metrics
from quantumfleet.prediction.features import build_feature_matrix, build_target, make_encoder, train_test_split_by_vessel
from quantumfleet.prediction.ml_model import FuelPredictionModel
from quantumfleet.reporting.charts import archive_to_dataframe, convergence_line_chart, emissions_by_fuel_bar, pareto_front_scatter_3d
from quantumfleet.reporting.report_builder import build_report, per_assignment_breakdown

st.set_page_config(page_title="Quantum-Inspired Green Fleet Optimizer", layout="wide")

SCENARIOS = {
    "Default scenario (8 routes)": PROJECT_ROOT / "configs" / "default_scenario.yaml",
    "Demonstration case study (50 routes)": PROJECT_ROOT / "configs" / "demo_case_study.yaml",
}
VESSEL_CATALOG_PATH = PROJECT_ROOT / "configs" / "vessel_types.yaml"


@st.cache_data
def _load_problem_cached(scenario_path: str, carbon_price: float, green_pathway: bool) -> ProblemSpec:
    from dataclasses import replace

    problem = load_problem(scenario_path, str(VESSEL_CATALOG_PATH))
    scenario = replace(problem.scenario, carbon_price_usd_per_tonne=carbon_price, green_fuel_pathway=green_pathway)
    return replace(problem, scenario=scenario)


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

with st.sidebar:
    st.header("Scenario")
    scenario_label = st.selectbox("Scenario", list(SCENARIOS.keys()))
    carbon_price = st.number_input("Carbon price (USD/t CO2e)", min_value=0.0, max_value=500.0, value=50.0, step=10.0)
    green_pathway = st.checkbox("Use green (low-carbon) production pathway for alt fuels", value=False)

    st.header("Quantum-inspired optimizer")
    population_size = st.slider("Population size", 10, 100, 40)
    n_generations = st.slider("Generations", 10, 200, 50)
    seed = st.number_input("Random seed", min_value=0, value=42, step=1)
    run_clicked = st.button("Run optimization", type="primary")

problem = _load_problem_cached(str(SCENARIOS[scenario_label]), carbon_price, green_pathway)

if run_clicked:
    with st.spinner(f"Running quantum-inspired optimizer on {len(problem.scenario.routes)} routes..."):
        optimizer = QuantumEvolutionaryOptimizer(problem=problem, population_size=population_size, n_generations=n_generations, seed=int(seed))
        archive, history = optimizer.run()
    st.session_state["archive"] = archive
    st.session_state["history"] = history
    st.session_state["problem"] = problem

tab_results, tab_scenario, tab_prediction, tab_benchmark, tab_about = st.tabs(
    ["Optimization Results", "Scenario & Routes", "Prediction Model", "Benchmarking", "Methodology"]
)

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

        st.plotly_chart(pareto_front_scatter_3d(archive), width="stretch")
        st.plotly_chart(convergence_line_chart({"Quantum-Inspired (QEA)": history}), width="stretch")

        st.subheader("Pick a plan by priority")
        wc1, wc2, wc3 = st.columns(3)
        w_fuel = wc1.slider("Fuel importance", 0.0, 1.0, 0.34)
        w_co2e = wc2.slider("Emissions importance", 0.0, 1.0, 0.33)
        w_cost = wc3.slider("Cost importance", 0.0, 1.0, 0.33)

        selected = _weighted_best_entry(archive, w_fuel, w_co2e, w_cost)
        o = selected.objectives
        st.write(f"Selected plan -- fuel **{o.fuel_tonnes:,.1f} t**, CO2e **{o.co2e_tonnes:,.1f} t**, "
                 f"cost **${o.cost_usd:,.0f}**, violation **{o.violation:.4f}**"
                 f"{' (infeasible)' if o.violation > 1e-9 else ''}")

        plan_df = per_assignment_breakdown(selected.plan, run_problem)
        st.dataframe(plan_df, width="stretch")
        if not plan_df.empty:
            st.plotly_chart(emissions_by_fuel_bar(plan_df), width="stretch")

        if st.button("Export report"):
            report_dir = PROJECT_ROOT / "results" / "dashboard_export"
            report_path = build_report(run_problem, archive, history, str(report_dir))
            st.success(f"Report written to {report_path}")
            st.download_button("Download report.md", data=Path(report_path).read_text(encoding="utf-8"), file_name="report.md")

with tab_scenario:
    st.subheader(f"Routes in '{problem.scenario.name}'")
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

            from quantumfleet.benchmarking.baselines_prediction import physics_baseline_predictions

            results = {
                "Physics (Admiralty)": regression_metrics(y_test, physics_baseline_predictions(test_df, problem.vessel_classes)),
            }
            for kind, label in [("linear", "Linear Regression"), ("random_forest", "Random Forest"), ("gradient_boosting", "Gradient Boosting")]:
                model = FuelPredictionModel(kind=kind).fit(X_train, y_train)
                results[label] = regression_metrics(y_test, model.predict(X_test))

        st.session_state["prediction_results"] = compare_models(results)

    if "prediction_results" in st.session_state:
        st.dataframe(st.session_state["prediction_results"].style.format("{:.4f}"), width="stretch")

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
