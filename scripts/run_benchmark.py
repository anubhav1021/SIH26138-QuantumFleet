"""CLI: benchmark the quantum-inspired optimizer against classical baselines
(random search, greedy heuristic, classical GA) across multiple route-count
instances, saving a comparison table and convergence charts.

Usage:
    python scripts/run_benchmark.py [--route-sizes 10 25 50] [--population-size N] [--generations N] [--repeats N]

Pass --repeats > 1 for a statistically meaningful comparison (mean/std
across independent runs, since these are stochastic algorithms) -- the
default of 1 run per algorithm/size is fast but each number is then just a
single sample, not a distribution.
"""

import argparse
import logging
from pathlib import Path

from quantumfleet.benchmarking.runner import run_benchmark_suite
from quantumfleet.optimization.problem import load_problem
from quantumfleet.reporting.charts import save_static_convergence
from quantumfleet.utils.io import save_dataframe
from quantumfleet.utils.logging_config import setup_logging

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default=str(PROJECT_ROOT / "configs" / "default_scenario.yaml"))
    parser.add_argument("--vessel-catalog", default=str(PROJECT_ROOT / "configs" / "vessel_types.yaml"))
    parser.add_argument("--route-sizes", type=int, nargs="+", default=[10, 25, 50])
    parser.add_argument("--population-size", type=int, default=40)
    parser.add_argument("--generations", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repeats", type=int, default=1, help="independent runs per algorithm/size, for mean/std (default 1 -- fast, but each number is a single sample)")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "results" / "benchmark"))
    args = parser.parse_args()

    setup_logging()
    log = logging.getLogger("run_benchmark")

    problem = load_problem(args.scenario, args.vessel_catalog)
    log.info("Benchmarking at route sizes %s (population=%d, generations=%d, repeats=%d)...", args.route_sizes, args.population_size, args.generations, args.repeats)

    result = run_benchmark_suite(problem, route_sizes=args.route_sizes, population_size=args.population_size, n_generations=args.generations, seed=args.seed, n_repeats=args.repeats)
    table, summary_table = result["table"], result["summary_table"]

    print("\nBenchmark comparison (per run):\n")
    print(table.to_string(index=False))
    if args.repeats > 1:
        print(f"\nSummary across {args.repeats} repeats (mean +/- std):\n")
        print(summary_table.to_string(index=False, float_format=lambda v: f"{v:.2f}"))

    out_dir = Path(args.output_dir)
    save_dataframe(table, out_dir / "benchmark_results.csv")
    save_dataframe(summary_table, out_dir / "benchmark_summary.csv")
    log.info("Saved comparison table to %s and summary to %s", out_dir / "benchmark_results.csv", out_dir / "benchmark_summary.csv")

    for n_routes in args.route_sizes:
        histories_at_size = {name: hist for (name, size), hist in result["convergence_curves"].items() if size == n_routes}
        chart_path = out_dir / f"convergence_{n_routes}routes.png"
        save_static_convergence(histories_at_size, str(chart_path))
        log.info("Saved convergence chart for %d routes to %s", n_routes, chart_path)


if __name__ == "__main__":
    main()
