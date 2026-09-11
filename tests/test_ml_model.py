import numpy as np

from quantumfleet.benchmarking.baselines_prediction import physics_baseline_predictions
from quantumfleet.data_generation.generator import generate_voyage_records
from quantumfleet.prediction.evaluate import compare_models, regression_metrics
from quantumfleet.prediction.features import build_feature_matrix, build_target, make_encoder, train_test_split_by_vessel
from quantumfleet.prediction.ml_model import FuelPredictionModel
from quantumfleet.prediction.physics_model import load_vessel_catalog


def _prepare(vessel_catalog_path, n_vessels=150, legs_per_vessel=20, seed=42):
    vessel_classes, _ = load_vessel_catalog(vessel_catalog_path)
    df = generate_voyage_records(vessel_classes, n_vessels=n_vessels, legs_per_vessel=legs_per_vessel, seed=seed)
    train_df, test_df = train_test_split_by_vessel(df, test_size=0.2, seed=seed)
    encoder = make_encoder(list(vessel_classes.keys()))
    return vessel_classes, train_df, test_df, encoder


def test_random_forest_achieves_high_r2_on_physics_generated_data(vessel_catalog_path):
    _, train_df, test_df, encoder = _prepare(vessel_catalog_path)
    X_train = build_feature_matrix(train_df, encoder)
    y_train = build_target(train_df)
    X_test = build_feature_matrix(test_df, encoder)
    y_test = build_target(test_df)

    model = FuelPredictionModel(kind="random_forest").fit(X_train, y_train)
    metrics = regression_metrics(y_test, model.predict(X_test))
    assert metrics["R2"] > 0.85


def test_random_forest_beats_linear_regression(vessel_catalog_path):
    _, train_df, test_df, encoder = _prepare(vessel_catalog_path)
    X_train = build_feature_matrix(train_df, encoder)
    y_train = build_target(train_df)
    X_test = build_feature_matrix(test_df, encoder)
    y_test = build_target(test_df)

    rf = FuelPredictionModel(kind="random_forest").fit(X_train, y_train)
    linear = FuelPredictionModel(kind="linear").fit(X_train, y_train)

    rf_r2 = regression_metrics(y_test, rf.predict(X_test))["R2"]
    linear_r2 = regression_metrics(y_test, linear.predict(X_test))["R2"]
    assert rf_r2 > linear_r2


def test_physics_baseline_matches_actual_closely(vessel_catalog_path):
    vessel_classes, _, test_df, _ = _prepare(vessel_catalog_path)
    y_test = build_target(test_df)
    physics_pred = physics_baseline_predictions(test_df, vessel_classes)
    metrics = regression_metrics(y_test, physics_pred)
    assert metrics["R2"] > 0.9


def test_compare_models_table_has_expected_rows():
    table = compare_models(
        {
            "physics": {"MAE": 1.0, "RMSE": 1.5, "MAPE": 0.05, "R2": 0.95},
            "linear": {"MAE": 3.0, "RMSE": 4.0, "MAPE": 0.15, "R2": 0.7},
        }
    )
    assert set(table.index) == {"physics", "linear"}
    assert "R2" in table.columns


def test_model_save_and_load_round_trip(tmp_path, vessel_catalog_path):
    _, train_df, test_df, encoder = _prepare(vessel_catalog_path, n_vessels=30, legs_per_vessel=10)
    X_train = build_feature_matrix(train_df, encoder)
    y_train = build_target(train_df)
    model = FuelPredictionModel(kind="random_forest").fit(X_train, y_train)

    save_path = tmp_path / "model.joblib"
    model.save(str(save_path))
    loaded = FuelPredictionModel.load(str(save_path))

    X_test = build_feature_matrix(test_df, encoder)
    np.testing.assert_allclose(model.predict(X_test), loaded.predict(X_test))


def test_encoder_handles_category_missing_from_a_sample(vessel_catalog_path):
    vessel_classes, _ = load_vessel_catalog(vessel_catalog_path)
    encoder = make_encoder(list(vessel_classes.keys()))
    # a fixed-category encoder must still produce the full-width feature
    # space even for a dataframe containing only a fraction of the classes.
    import pandas as pd

    small_df = pd.DataFrame(
        {
            "vessel_class": [next(iter(vessel_classes))],
            "fuel_type": ["HFO"],
            "speed_knots": [14.0],
            "load_factor": [0.8],
            "wave_height_m": [1.0],
            "wind_speed_knots": [5.0],
        }
    )
    X = build_feature_matrix(small_df, encoder)
    n_vessel_classes = len(vessel_classes)
    n_fuels = 6
    assert X.shape == (1, 4 + n_vessel_classes + n_fuels)
