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
    n_repeats: int = 1,
) -> dict:
    """Runs the QEA, a classical GA (same encoding/repair/constraints/pareto),
    and random search at each size in `route_sizes` under an identical
    evaluation budget, plus the one-shot greedy heuristic. Records runtime,
    final hypervolume, and generations-to-95%-hypervolume per run.

    `n_repeats` runs each stochastic algorithm that many times (different
    seeds, `seed + repeat_index`) since these are stochastic methods -- a
    single run can be lucky or unlucky. Default is 1 for backward
    compatibility (existing callers get exactly the old one-row-per-
    algorithm-per-size table); `result["summary_table"]` is always present
    and reports mean/std, honestly showing NaN std when n_repeats=1 rather
    than a false sense of only one number mattering. The greedy heuristic is
    deterministic, so it always runs once regardless of `n_repeats`.
    Convergence curves are kept only for the first repeat of each
    algorithm/size -- later repeats would clutter a chart without adding
    information the std-dev columns don't already summarize."""
    rows = []
    convergence_curves: dict[tuple[str, int], list[GenerationStats]] = {}

    for n_routes in route_sizes:
        instance = scaled_problem(problem, n_routes)

        for repeat in range(n_repeats):
            run_seed = seed + repeat
            stochastic_algorithms = {
                "Quantum-Inspired (QEA)": lambda s=run_seed: QuantumEvolutionaryOptimizer(problem=instance, population_size=population_size, n_generations=n_generations, seed=s).run(),
                "Classical GA": lambda s=run_seed: ClassicalGeneticAlgorithm(problem=instance, population_size=population_size, n_generations=n_generations, seed=s).run(),
                "Random Search": lambda s=run_seed: run_random_search(instance, population_size=population_size, n_generations=n_generations, seed=s),
            }

            for name, run_fn in stochastic_algorithms.items():
                start = time.perf_counter()
                archive, history = run_fn()
                elapsed = time.perf_counter() - start

                rows.append(
                    {
                        "algorithm": name,
                        "n_routes": n_routes,
                        "repeat": repeat,
                        "seed": run_seed,
                        "runtime_sec": elapsed,
                        "final_hypervolume": history[-1].hypervolume if history else 0.0,
                        "generations_to_95pct_hv": generations_to_reach_hypervolume(history),
                        "archive_size": len(archive),
                        "feasible_count": sum(1 for e in archive.entries if e.objectives.violation <= 1e-9),
                    }
                )
                if repeat == 0:
                    convergence_curves[(name, n_routes)] = history

        start = time.perf_counter()
        greedy_plan = run_greedy_heuristic(instance)
        elapsed = time.perf_counter() - start
        greedy_obj = evaluate(greedy_plan, instance)
        rows.append(
            {
                "algorithm": "Greedy Heuristic",
                "n_routes": n_routes,
                "repeat": 0,
                "seed": seed,
                "runtime_sec": elapsed,
                "final_hypervolume": None,
                "generations_to_95pct_hv": None,
                "archive_size": 1,
                "feasible_count": int(greedy_obj.violation <= 1e-9),
            }
        )

    table = pd.DataFrame(rows)
    return {"table": table, "summary_table": summarize_repeats(table), "convergence_curves": convergence_curves}


def summarize_repeats(raw_table: pd.DataFrame) -> pd.DataFrame:
    """Aggregates (possibly repeated) runs into mean/std per algorithm and
    route count. With a single repeat, std is NaN -- an honest signal that
    no variance estimate exists yet, not a hidden zero."""
    numeric_cols = ["runtime_sec", "final_hypervolume", "generations_to_95pct_hv", "archive_size", "feasible_count"]
    grouped = raw_table.groupby(["algorithm", "n_routes"])[numeric_cols]
    mean_df = grouped.mean().add_suffix("_mean")
    std_df = grouped.std().add_suffix("_std")
    n_runs = grouped.size().rename("n_runs")
    return pd.concat([mean_df, std_df, n_runs], axis=1).reset_index()
