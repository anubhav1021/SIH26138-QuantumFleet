from dataclasses import replace

import numpy as np

from quantumfleet.optimization.problem import load_problem
from quantumfleet.optimization.qea import QuantumEvolutionaryOptimizer


def _tiny_problem(vessel_catalog_path, default_scenario_path, n_routes=3):
    problem = load_problem(default_scenario_path, vessel_catalog_path)
    small_scenario = replace(problem.scenario, routes=problem.scenario.routes[:n_routes])
    return replace(problem, scenario=small_scenario)


def test_qea_produces_non_empty_non_dominated_archive(vessel_catalog_path, default_scenario_path):
    problem = _tiny_problem(vessel_catalog_path, default_scenario_path)
    optimizer = QuantumEvolutionaryOptimizer(problem=problem, population_size=20, n_generations=30, seed=1)
    archive, history = optimizer.run()

    assert len(archive) > 0
    assert archive.is_non_dominated_set()
    assert len(history) == 30


def test_qea_hypervolume_does_not_regress(vessel_catalog_path, default_scenario_path):
    problem = _tiny_problem(vessel_catalog_path, default_scenario_path)
    optimizer = QuantumEvolutionaryOptimizer(problem=problem, population_size=20, n_generations=40, seed=2)
    _, history = optimizer.run()

    early_avg = np.mean([h.hypervolume for h in history[:5]])
    late_avg = np.mean([h.hypervolume for h in history[-5:]])
    assert late_avg >= early_avg


def test_qea_finds_feasible_solutions(vessel_catalog_path, default_scenario_path):
    problem = _tiny_problem(vessel_catalog_path, default_scenario_path)
    optimizer = QuantumEvolutionaryOptimizer(problem=problem, population_size=30, n_generations=60, seed=3)
    archive, _ = optimizer.run()

    feasible = [e for e in archive.entries if e.objectives.violation <= 1e-9]
    assert len(feasible) > 0


def test_qea_is_reproducible_with_same_seed(vessel_catalog_path, default_scenario_path):
    problem = _tiny_problem(vessel_catalog_path, default_scenario_path, n_routes=2)
    archive1, _ = QuantumEvolutionaryOptimizer(problem=problem, population_size=10, n_generations=10, seed=5).run()
    archive2, _ = QuantumEvolutionaryOptimizer(problem=problem, population_size=10, n_generations=10, seed=5).run()

    fuels1 = sorted(e.objectives.fuel_tonnes for e in archive1.entries)
    fuels2 = sorted(e.objectives.fuel_tonnes for e in archive2.entries)
    assert fuels1 == fuels2
