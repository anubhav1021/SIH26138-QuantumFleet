"""CLI: the Deliverable-5 demonstration. Runs the quantum-inspired optimizer
on the full 50-route demo case study, then benchmarks it against classical
baselines at a couple of smaller scales, and writes everything to
docs/case_study_results.md.

Usage:
    python scripts/run_case_study.py [--population-size N] [--generations N]
"""

import argparse
import logging
from pathlib import Path

from quantumfleet.benchmarking.metrics import generations_to_reach_hypervolume
from quantumfleet.benchmarking.runner import run_benchmark_suite
from quantumfleet.optimization.problem import load_problem
from quantumfleet.optimization.qea import QuantumEvolutionaryOptimizer
from quantumfleet.reporting.charts import save_static_convergence
from quantumfleet.reporting.report_builder import build_report
from quantumfleet.utils.logging_config import setup_logging

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default=str(PROJECT_ROOT / "configs" / "demo_case_study.yaml"))
    parser.add_argument("--vessel-catalog", default=str(PROJECT_ROOT / "configs" / "vessel_types.yaml"))
    parser.add_argument("--population-size", type=int, default=40)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--benchmark-route-sizes", type=int, nargs="+", default=[10, 25])
    parser.add_argument("--benchmark-generations", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "docs"))
    args = parser.parse_args()

    setup_logging()
    log = logging.getLogger("run_case_study")

    problem = load_problem(args.scenario, args.vessel_catalog)
    log.info("Loaded '%s': %d routes.", problem.scenario.name, len(problem.scenario.routes))

    log.info("Running QEA on the full case study: population=%d generations=%d...", args.population_size, args.generations)
    optimizer = QuantumEvolutionaryOptimizer(problem=problem, population_size=args.population_size, n_generations=args.generations, seed=args.seed)
    archive, history = optimizer.run()

    feasible = [e for e in archive.entries if e.objectives.violation <= 1e-9]
    log.info("Archive: %d non-dominated plans, %d fully feasible. Final hypervolume: %.1f", len(archive), len(feasible), history[-1].hypervolume)

    report_path = build_report(problem, archive, history, args.output_dir, title="SIH26138 Demonstration: 50-Route Case Study", filename="case_study_results.md", image_prefix="case_study_")
    log.info("Wrote main report to %s", report_path)

    log.info("Benchmarking against classical baselines at route sizes %s...", args.benchmark_route_sizes)
    bench = run_benchmark_suite(problem, route_sizes=args.benchmark_route_sizes, population_size=args.population_size, n_generations=args.benchmark_generations, seed=args.seed)
    table = bench["table"]

    for n_routes in args.benchmark_route_sizes:
        histories_at_size = {name: hist for (name, size), hist in bench["convergence_curves"].items() if size == n_routes}
        save_static_convergence(histories_at_size, str(Path(args.output_dir) / f"case_study_benchmark_convergence_{n_routes}routes.png"))

    benchmark_section = ["", "## Benchmarking: quantum-inspired vs. classical baselines", ""]
    for n_routes in args.benchmark_route_sizes:
        subset = table[table["n_routes"] == n_routes].drop(columns=["n_routes"])
        benchmark_section += [f"### {n_routes} routes", ""]
        header = "| " + " | ".join(subset.columns) + " |"
        separator = "| " + " | ".join("---" for _ in subset.columns) + " |"
        rows = ["| " + " | ".join(f"{v:.1f}" if isinstance(v, float) else str(v) for v in row) + " |" for row in subset.itertuples(index=False)]
        benchmark_section += [header, separator, *rows, "", f"![Convergence at {n_routes} routes](case_study_benchmark_convergence_{n_routes}routes.png)", ""]

    with open(report_path, "a", encoding="utf-8") as f:
        f.write("\n".join(benchmark_section))

    log.info("Appended benchmarking section to %s", report_path)
    print(f"\nDone. Full demonstration report: {report_path}")


if __name__ == "__main__":
    main()
