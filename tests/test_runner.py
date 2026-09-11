from quantumfleet.benchmarking.runner import run_benchmark_suite, scaled_problem
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
