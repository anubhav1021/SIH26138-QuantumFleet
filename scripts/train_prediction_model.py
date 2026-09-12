"""CLI: generate synthetic voyage data, then train and evaluate the fuel-
consumption prediction models (physics baseline, linear regression, random
forest, gradient boosting, and a QPSO-tuned random forest).

Usage:
    python scripts/train_prediction_model.py [--n-vessels N] [--legs-per-vessel N] [--seed N]
"""

import argparse
import logging
from pathlib import Path

from quantumfleet.benchmarking.baselines_prediction import physics_baseline_predictions
from quantumfleet.data_generation.generator import generate_voyage_records
from quantumfleet.data_generation.validate import assert_valid_voyage_records
from quantumfleet.prediction.evaluate import compare_models, regression_metrics
from quantumfleet.prediction.features import build_feature_matrix, build_target, make_encoder, train_test_split_by_vessel
from quantumfleet.prediction.hybrid_model import HybridFuelPredictionModel, residual_learnability_r2
from quantumfleet.prediction.ml_model import FuelPredictionModel
from quantumfleet.prediction.physics_model import load_vessel_catalog
from quantumfleet.prediction.qpso_tuner import DEFAULT_RF_BOUNDS, quantum_pso_tune, rf_cv_rmse_objective
from quantumfleet.utils.io import save_dataframe
from quantumfleet.utils.logging_config import setup_logging

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-vessels", type=int, default=200)
    parser.add_argument("--legs-per-vessel", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-qpso", action="store_true", help="skip QPSO hyperparameter tuning (it's the slowest step)")
    parser.add_argument("--vessel-catalog", default=str(PROJECT_ROOT / "configs" / "vessel_types.yaml"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "data" / "synthetic"))
    parser.add_argument("--model-dir", default=str(PROJECT_ROOT / "results" / "models"))
    args = parser.parse_args()

    setup_logging()
    log = logging.getLogger("train_prediction_model")

    vessel_classes, _ = load_vessel_catalog(args.vessel_catalog)
    encoder = make_encoder(list(vessel_classes.keys()))

    log.info("Generating %d vessels x %d legs of synthetic voyage records...", args.n_vessels, args.legs_per_vessel)
    df = generate_voyage_records(vessel_classes, n_vessels=args.n_vessels, legs_per_vessel=args.legs_per_vessel, seed=args.seed)
    assert_valid_voyage_records(df)
    save_dataframe(df, Path(args.output_dir) / "voyage_records.csv")
    log.info("Generated %d records -> %s", len(df), Path(args.output_dir) / "voyage_records.csv")

    train_df, test_df = train_test_split_by_vessel(df, test_size=0.2, seed=args.seed)
    X_train = build_feature_matrix(train_df, encoder)
    y_train = build_target(train_df)
    X_test = build_feature_matrix(test_df, encoder)
    y_test = build_target(test_df)

    results: dict[str, dict[str, float]] = {}

    log.info("Evaluating physics-only baseline...")
    physics_pred = physics_baseline_predictions(test_df, vessel_classes)
    results["Physics (Admiralty)"] = regression_metrics(y_test, physics_pred)

    log.info("Training linear regression baseline...")
    linear = FuelPredictionModel(kind="linear").fit(X_train, y_train)
    results["Linear Regression"] = regression_metrics(y_test, linear.predict(X_test))

    log.info("Training random forest model (default hyperparameters)...")
    rf = FuelPredictionModel(kind="random_forest").fit(X_train, y_train)
    results["Random Forest"] = regression_metrics(y_test, rf.predict(X_test))

    log.info("Training gradient boosting model...")
    gb = FuelPredictionModel(kind="gradient_boosting").fit(X_train, y_train)
    results["Gradient Boosting"] = regression_metrics(y_test, gb.predict(X_test))

    best_model = rf
    if not args.skip_qpso:
        log.info("Tuning random forest hyperparameters with quantum-behaved PSO...")
        objective = rf_cv_rmse_objective(X_train, y_train, cv=3)
        tuning = quantum_pso_tune(objective, bounds=DEFAULT_RF_BOUNDS, n_particles=12, n_iterations=15, seed=args.seed)
        log.info("QPSO best params: %s (CV RMSE=%.3f)", tuning.best_params, tuning.best_score)
        qpso_rf = FuelPredictionModel(kind="random_forest", hyperparams=tuning.best_params).fit(X_train, y_train)
        results["Random Forest + QPSO"] = regression_metrics(y_test, qpso_rf.predict(X_test))
        if results["Random Forest + QPSO"]["RMSE"] < results["Random Forest"]["RMSE"]:
            best_model = qpso_rf

    log.info("Training hybrid physics+ML residual-correction model...")
    hybrid = HybridFuelPredictionModel().fit(train_df, X_train, vessel_classes)
    results["Hybrid (physics + RF residual)"] = regression_metrics(y_test, hybrid.predict(test_df, X_test, vessel_classes))
    residual_r2 = residual_learnability_r2(train_df, X_train, vessel_classes)
    log.info(
        "Residual learnability (CV R2 of predicting actual-physics from features): %.3f -- %s",
        residual_r2,
        "near zero/negative: little systematic bias for the hybrid model to correct on this synthetic data"
        if residual_r2 < 0.1
        else "meaningfully positive: the hybrid model is capturing a real systematic bias",
    )

    table = compare_models(results)
    print("\nPrediction accuracy comparison (held-out test set):\n")
    print(table.to_string(float_format=lambda v: f"{v:.4f}"))

    best_model.save(str(Path(args.model_dir) / "random_forest.joblib"))
    log.info("Saved best-performing model to %s", args.model_dir)


if __name__ == "__main__":
    main()
