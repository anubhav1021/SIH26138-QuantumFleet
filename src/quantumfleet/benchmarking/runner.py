"""Orchestrates running the QEA against classical baselines across multiple
instance sizes with a shared evaluation budget, producing the raw comparison
tables and convergence curves consumed by `reporting`.
"""

import time
from dataclasses import replace

import pandas as pd

from quantumfleet.benchmarking.baselines_optimization import ClassicalGeneticAlgorithm, run_greedy_heuristic, run_random_search
from quantumfleet.benchmarking.metrics import generations_to_reach_hypervolume
from quantumfleet.optimization.problem import ProblemSpec
from quantumfleet.optimization.qea import GenerationStats, QuantumEvolutionaryOptimizer, evaluate


def scaled_problem(problem: ProblemSpec, n_routes: int) -> ProblemSpec:
    """Builds an n_routes-route instance by cycling (with relabelled, unique
    IDs) through the base scenario's routes -- lets the scalability sweep go
    beyond however many routes are actually defined in the scenario config
    while keeping each route's own demand/schedule/emission profile
    realistic rather than synthesizing arbitrary new ones."""
    base_routes = problem.scenario.routes
    routes = tuple(replace(base_routes[i % len(base_routes)], route_id=f"{base_routes[i % len(base_routes)].route_id}__{i}") for i in range(n_routes))
    return replace(problem, scenario=replace(problem.scenario, routes=routes))


def run_benchmark_suite(
    problem: ProblemSpec,
    route_sizes: list[int] = [10, 25, 50],
    population_size: int = 40,
    n_generations: int = 100,
    seed: int = 42,
) -> dict:
    """Runs the QEA, a classical GA (same encoding/repair/constraints/pareto),
    and random search at each size in `route_sizes` under an identical
    evaluation budget, plus the one-shot greedy heuristic. Records runtime,
    final hypervolume, and generations-to-95%-hypervolume per run."""
    rows = []
    convergence_curves: dict[tuple[str, int], list[GenerationStats]] = {}

    for n_routes in route_sizes:
        instance = scaled_problem(problem, n_routes)

        stochastic_algorithms = {
            "Quantum-Inspired (QEA)": lambda: QuantumEvolutionaryOptimizer(problem=instance, population_size=population_size, n_generations=n_generations, seed=seed).run(),
            "Classical GA": lambda: ClassicalGeneticAlgorithm(problem=instance, population_size=population_size, n_generations=n_generations, seed=seed).run(),
            "Random Search": lambda: run_random_search(instance, population_size=population_size, n_generations=n_generations, seed=seed),
        }

        for name, run_fn in stochastic_algorithms.items():
            start = time.perf_counter()
            archive, history = run_fn()
            elapsed = time.perf_counter() - start

            rows.append(
                {
                    "algorithm": name,
                    "n_routes": n_routes,
                    "runtime_sec": elapsed,
                    "final_hypervolume": history[-1].hypervolume if history else 0.0,
                    "generations_to_95pct_hv": generations_to_reach_hypervolume(history),
                    "archive_size": len(archive),
                    "feasible_count": sum(1 for e in archive.entries if e.objectives.violation <= 1e-9),
                }
            )
            convergence_curves[(name, n_routes)] = history

        start = time.perf_counter()
        greedy_plan = run_greedy_heuristic(instance)
        elapsed = time.perf_counter() - start
        greedy_obj = evaluate(greedy_plan, instance)
        rows.append(
            {
                "algorithm": "Greedy Heuristic",
                "n_routes": n_routes,
                "runtime_sec": elapsed,
                "final_hypervolume": None,
                "generations_to_95pct_hv": None,
                "archive_size": 1,
                "feasible_count": int(greedy_obj.violation <= 1e-9),
            }
        )

    return {"table": pd.DataFrame(rows), "convergence_curves": convergence_curves}
