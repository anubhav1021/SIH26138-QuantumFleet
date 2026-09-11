"""CLI: run the quantum-inspired fleet optimizer on a scenario and report the
resulting Pareto front, plus a physics-vs-trained-ML cross-check on the
lowest-fuel plan found.

Usage:
    python scripts/run_optimization.py [--scenario PATH] [--population-size N] [--generations N]
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from quantumfleet.data_generation.schema import VoyageRecordColumns as C
from quantumfleet.optimization.problem import load_problem
from quantumfleet.optimization.qea import DEFAULT_LOAD_FACTOR_ASSUMPTION, QuantumEvolutionaryOptimizer
from quantumfleet.prediction.features import build_feature_matrix, make_encoder
from quantumfleet.prediction.ml_model import FuelPredictionModel
from quantumfleet.prediction.physics_model import VoyageConditions, predict_fuel_consumption_physics
from quantumfleet.utils.logging_config import setup_logging

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", default=str(PROJECT_ROOT / "configs" / "default_scenario.yaml"))
    parser.add_argument("--vessel-catalog", default=str(PROJECT_ROOT / "configs" / "vessel_types.yaml"))
    parser.add_argument("--population-size", type=int, default=40)
    parser.add_argument("--generations", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ml-model", default=str(PROJECT_ROOT / "results" / "models" / "random_forest.joblib"))
    args = parser.parse_args()

    setup_logging()
    log = logging.getLogger("run_optimization")

    problem = load_problem(args.scenario, args.vessel_catalog)
    log.info("Loaded scenario '%s' with %d routes.", problem.scenario.name, len(problem.scenario.routes))

    optimizer = QuantumEvolutionaryOptimizer(problem=problem, population_size=args.population_size, n_generations=args.generations, seed=args.seed)
    log.info("Running QEA: population=%d generations=%d...", args.population_size, args.generations)
    archive, history = optimizer.run()

    feasible = [e for e in archive.entries if e.objectives.violation <= 1e-9]
    print(f"\nPareto archive: {len(archive)} non-dominated plans ({len(feasible)} fully feasible).")
    print(f"Final hypervolume: {history[-1].hypervolume:,.1f}\n")

    if not feasible:
        print("No fully feasible plan found -- showing the least-violating plan instead.\n")
        feasible = sorted(archive.entries, key=lambda e: e.objectives.violation)[:1]

    print(f"{'fuel_tonnes':>14} {'co2e_tonnes':>14} {'cost_usd':>16} {'violation':>10}")
    for e in sorted(feasible, key=lambda e: e.objectives.cost_usd)[:10]:
        o = e.objectives
        print(f"{o.fuel_tonnes:14,.1f} {o.co2e_tonnes:14,.1f} {o.cost_usd:16,.0f} {o.violation:10.4f}")

    best = min(feasible, key=lambda e: e.objectives.fuel_tonnes)
    print(f"\nLowest-fuel feasible plan ({best.objectives.fuel_tonnes:,.1f}t) -- active assignments:")
    for a in best.plan.assignments:
        if a.count > 0:
            print(f"  {a.route_id}: {a.count}x {a.vessel_class} @ {a.speed_knots}kn on {a.fuel_type}"
                  f"{' + shore power' if a.use_shore_power else ''}")

    if Path(args.ml_model).exists():
        log.info("Cross-checking best plan against the trained ML prediction model...")
        model = FuelPredictionModel.load(args.ml_model)
        encoder = make_encoder(list(problem.vessel_classes.keys()))

        active = [a for a in best.plan.assignments if a.count > 0]
        rows = pd.DataFrame(
            {
                C.VESSEL_CLASS: [a.vessel_class for a in active],
                C.FUEL_TYPE: [a.fuel_type for a in active],
                C.SPEED_KNOTS: [a.speed_knots for a in active],
                C.LOAD_FACTOR: [DEFAULT_LOAD_FACTOR_ASSUMPTION] * len(active),
                C.WAVE_HEIGHT_M: [0.0] * len(active),
                C.WIND_SPEED_KNOTS: [0.0] * len(active),
            }
        )
        ml_daily_tonnes = model.predict(build_feature_matrix(rows, encoder))

        print("\nPhysics vs. trained-ML daily fuel-rate cross-check (calm-weather, same speed/load/fuel as the plan):")
        print(f"{'route_id':<20} {'vessel_class':<20} {'fuel':<10} {'physics_t/day':>14} {'ml_t/day':>10} {'diff':>8}")
        for a, ml_rate in zip(active, ml_daily_tonnes):
            vessel = problem.vessel_classes[a.vessel_class]
            conditions = VoyageConditions(speed_knots=a.speed_knots, load_factor=DEFAULT_LOAD_FACTOR_ASSUMPTION, distance_nm=a.speed_knots * 24.0)
            physics_rate = predict_fuel_consumption_physics(vessel, conditions, a.fuel_type).fuel_tonnes
            diff_pct = 100.0 * (ml_rate - physics_rate) / physics_rate if physics_rate else float("nan")
            print(f"{a.route_id:<20} {a.vessel_class:<20} {a.fuel_type:<10} {physics_rate:14.1f} {ml_rate:10.1f} {diff_pct:7.1f}%")
    else:
        log.info("No trained ML model found at %s (run scripts/train_prediction_model.py first) -- skipping cross-check.", args.ml_model)


if __name__ == "__main__":
    main()
