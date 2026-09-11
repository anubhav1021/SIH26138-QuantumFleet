"""Full-pipeline smoke test: data generation -> prediction model -> quantum-
inspired optimization -> reporting, all at a tiny scale. Complements the
per-module unit tests by confirming the pieces actually wire together."""

from dataclasses import replace
from pathlib import Path

from quantumfleet.data_generation.generator import generate_voyage_records
from quantumfleet.data_generation.validate import assert_valid_voyage_records
from quantumfleet.optimization.problem import load_problem
from quantumfleet.optimization.qea import QuantumEvolutionaryOptimizer
from quantumfleet.prediction.evaluate import regression_metrics
from quantumfleet.prediction.features import build_feature_matrix, build_target, make_encoder, train_test_split_by_vessel
from quantumfleet.prediction.ml_model import FuelPredictionModel
from quantumfleet.reporting.report_builder import build_report


def test_full_pipeline_runs_end_to_end(tmp_path, vessel_catalog_path, default_scenario_path):
    problem = load_problem(default_scenario_path, vessel_catalog_path)

    df = generate_voyage_records(problem.vessel_classes, n_vessels=20, legs_per_vessel=10, seed=1)
    assert_valid_voyage_records(df)

    train_df, test_df = train_test_split_by_vessel(df, test_size=0.2, seed=1)
    encoder = make_encoder(list(problem.vessel_classes.keys()))
    model = FuelPredictionModel(kind="random_forest").fit(build_feature_matrix(train_df, encoder), build_target(train_df))
    metrics = regression_metrics(build_target(test_df), model.predict(build_feature_matrix(test_df, encoder)))
    assert metrics["R2"] > 0.5  # loose bound: this test checks wiring, not accuracy (see test_ml_model.py for that)

    small_problem = replace(problem, scenario=replace(problem.scenario, routes=problem.scenario.routes[:2]))
    archive, history = QuantumEvolutionaryOptimizer(problem=small_problem, population_size=15, n_generations=20, seed=1).run()
    assert len(archive) > 0

    report_path = build_report(small_problem, archive, history, str(tmp_path))
    assert Path(report_path).exists()
    assert "Fleet allocation" in Path(report_path).read_text(encoding="utf-8")
