from dataclasses import replace

from quantumfleet.benchmarking.baselines_optimization import ClassicalGeneticAlgorithm, run_greedy_heuristic, run_random_search
from quantumfleet.optimization.problem import load_problem
from quantumfleet.optimization.qea import evaluate


def _tiny_problem(vessel_catalog_path, default_scenario_path, n_routes=3):
    problem = load_problem(default_scenario_path, vessel_catalog_path)
    return replace(problem, scenario=replace(problem.scenario, routes=problem.scenario.routes[:n_routes]))


def test_random_search_runs_and_produces_a_history(vessel_catalog_path, default_scenario_path):
    problem = _tiny_problem(vessel_catalog_path, default_scenario_path, n_routes=2)
    archive, history = run_random_search(problem, population_size=15, n_generations=10, seed=1)
    assert len(history) == 10
    assert len(archive) > 0


def test_greedy_heuristic_produces_one_assignment_per_route(vessel_catalog_path, default_scenario_path):
    problem = _tiny_problem(vessel_catalog_path, default_scenario_path, n_routes=3)
    plan = run_greedy_heuristic(problem)
    assert len(plan.assignments) == 3
    assert {a.route_id for a in plan.assignments} == {r.route_id for r in problem.scenario.routes}


def test_greedy_heuristic_is_deterministic(vessel_catalog_path, default_scenario_path):
    problem = _tiny_problem(vessel_catalog_path, default_scenario_path, n_routes=2)
    assert run_greedy_heuristic(problem) == run_greedy_heuristic(problem)


def test_greedy_heuristic_finds_a_feasible_plan(vessel_catalog_path, default_scenario_path):
    problem = _tiny_problem(vessel_catalog_path, default_scenario_path, n_routes=3)
    plan = run_greedy_heuristic(problem)
    assert evaluate(plan, problem).violation <= 1e-9


def test_classical_ga_produces_non_dominated_archive(vessel_catalog_path, default_scenario_path):
    problem = _tiny_problem(vessel_catalog_path, default_scenario_path, n_routes=2)
    archive, history = ClassicalGeneticAlgorithm(problem=problem, population_size=20, n_generations=30, seed=2).run()
    assert len(archive) > 0
    assert archive.is_non_dominated_set()
    assert len(history) == 30


def test_classical_ga_finds_feasible_solutions(vessel_catalog_path, default_scenario_path):
    problem = _tiny_problem(vessel_catalog_path, default_scenario_path, n_routes=3)
    archive, _ = ClassicalGeneticAlgorithm(problem=problem, population_size=30, n_generations=80, seed=1).run()
    feasible = [e for e in archive.entries if e.objectives.violation <= 1e-9]
    assert len(feasible) > 0


def test_classical_ga_is_reproducible_with_same_seed(vessel_catalog_path, default_scenario_path):
    problem = _tiny_problem(vessel_catalog_path, default_scenario_path, n_routes=1)
    a1, _ = ClassicalGeneticAlgorithm(problem=problem, population_size=10, n_generations=10, seed=9).run()
    a2, _ = ClassicalGeneticAlgorithm(problem=problem, population_size=10, n_generations=10, seed=9).run()
    assert sorted(e.objectives.fuel_tonnes for e in a1.entries) == sorted(e.objectives.fuel_tonnes for e in a2.entries)
