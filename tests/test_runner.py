import numpy as np

from quantumfleet.benchmarking.runner import run_benchmark_suite, scaled_problem, summarize_repeats
from quantumfleet.optimization.problem import load_problem


def test_scaled_problem_produces_requested_route_count(vessel_catalog_path, default_scenario_path):
    problem = load_problem(default_scenario_path, vessel_catalog_path)
    scaled = scaled_problem(problem, 20)
    assert len(scaled.scenario.routes) == 20
    assert len({r.route_id for r in scaled.scenario.routes}) == 20  # relabelled IDs stay unique


def test_run_benchmark_suite_produces_expected_table_shape(vessel_catalog_path, default_scenario_path):
    problem = load_problem(default_scenario_path, vessel_catalog_path)
    small_problem = scaled_problem(problem, 2)

    result = run_benchmark_suite(small_problem, route_sizes=[2, 4], population_size=10, n_generations=8, seed=1)
    table = result["table"]

    # 3 stochastic algorithms + 1 greedy row, per route size
    assert len(table) == 2 * (3 + 1)
    assert set(table["algorithm"].unique()) == {"Quantum-Inspired (QEA)", "Classical GA", "Random Search", "Greedy Heuristic"}
    assert set(table["n_routes"].unique()) == {2, 4}
    assert (table["runtime_sec"] >= 0).all()

    curves = result["convergence_curves"]
    assert ("Quantum-Inspired (QEA)", 2) in curves
    assert len(curves[("Quantum-Inspired (QEA)", 2)]) == 8

    # backward compatibility: default n_repeats=1 keeps exactly one row per algorithm/size
    assert (table.groupby(["algorithm", "n_routes"]).size() == 1).all()


def test_run_benchmark_suite_with_repeats_produces_multiple_rows_and_summary(vessel_catalog_path, default_scenario_path):
    problem = load_problem(default_scenario_path, vessel_catalog_path)
    small_problem = scaled_problem(problem, 2)

    result = run_benchmark_suite(small_problem, route_sizes=[2], population_size=10, n_generations=8, seed=1, n_repeats=3)
    table = result["table"]

    stochastic = table[table["algorithm"] != "Greedy Heuristic"]
    assert (stochastic.groupby("algorithm").size() == 3).all()
    assert set(stochastic["seed"].unique()) == {1, 2, 3}

    greedy = table[table["algorithm"] == "Greedy Heuristic"]
    assert len(greedy) == 1  # deterministic: always one row regardless of n_repeats

    summary = result["summary_table"]
    assert "final_hypervolume_mean" in summary.columns
    assert "final_hypervolume_std" in summary.columns
    qea_summary = summary[summary["algorithm"] == "Quantum-Inspired (QEA)"].iloc[0]
    assert qea_summary["n_runs"] == 3


def test_summarize_repeats_std_is_nan_for_single_run():
    import pandas as pd

    raw = pd.DataFrame(
        [{"algorithm": "X", "n_routes": 5, "runtime_sec": 1.0, "final_hypervolume": 10.0, "generations_to_95pct_hv": 3, "archive_size": 2, "feasible_count": 1}]
    )
    summary = summarize_repeats(raw)
    assert summary.loc[0, "n_runs"] == 1
    assert np.isnan(summary.loc[0, "final_hypervolume_std"])
