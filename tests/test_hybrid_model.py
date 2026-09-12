from quantumfleet.benchmarking.baselines_prediction import physics_baseline_predictions
from quantumfleet.data_generation.generator import generate_voyage_records
from quantumfleet.prediction.evaluate import regression_metrics
from quantumfleet.prediction.features import build_feature_matrix, build_target, make_encoder, train_test_split_by_vessel
from quantumfleet.prediction.hybrid_model import HybridFuelPredictionModel, residual_learnability_r2
from quantumfleet.prediction.physics_model import load_vessel_catalog


def _prepare(vessel_catalog_path, n_vessels=150, legs_per_vessel=20, seed=42):
    vessel_classes, _ = load_vessel_catalog(vessel_catalog_path)
    df = generate_voyage_records(vessel_classes, n_vessels=n_vessels, legs_per_vessel=legs_per_vessel, seed=seed)
    train_df, test_df = train_test_split_by_vessel(df, test_size=0.2, seed=seed)
    encoder = make_encoder(list(vessel_classes.keys()))
    return vessel_classes, train_df, test_df, encoder


def test_hybrid_model_fits_and_predicts_reasonable_values(vessel_catalog_path):
    vessel_classes, train_df, test_df, encoder = _prepare(vessel_catalog_path)
    X_train = build_feature_matrix(train_df, encoder)
    X_test = build_feature_matrix(test_df, encoder)

    model = HybridFuelPredictionModel().fit(train_df, X_train, vessel_classes)
    predictions = model.predict(test_df, X_test, vessel_classes)

    assert len(predictions) == len(test_df)
    assert (predictions > 0).all()  # fuel consumption can't be negative or zero for a real voyage


def test_hybrid_model_raises_if_not_fitted(vessel_catalog_path):
    vessel_classes, _, test_df, encoder = _prepare(vessel_catalog_path)
    X_test = build_feature_matrix(test_df, encoder)
    model = HybridFuelPredictionModel()
    try:
        model.predict(test_df, X_test, vessel_classes)
        assert False, "expected RuntimeError"
    except RuntimeError:
        pass


def test_hybrid_model_performs_comparably_to_physics_on_synthetic_data(vessel_catalog_path):
    """Locks in the honest finding documented in hybrid_model.py: this
    platform's synthetic data has no feature-learnable residual bias (a
    per-vessel random offset uncorrelated with any observed feature, plus
    i.i.d. noise), so the hybrid model should land close to physics alone --
    not dramatically better, and not badly worse either. A test that only
    allowed "hybrid must beat physics" would encode a claim the data doesn't
    support; this instead locks in the honest comparison."""
    vessel_classes, train_df, test_df, encoder = _prepare(vessel_catalog_path)
    X_train = build_feature_matrix(train_df, encoder)
    X_test = build_feature_matrix(test_df, encoder)
    y_test = build_target(test_df)

    physics_r2 = regression_metrics(y_test, physics_baseline_predictions(test_df, vessel_classes))["R2"]

    hybrid = HybridFuelPredictionModel().fit(train_df, X_train, vessel_classes)
    hybrid_r2 = regression_metrics(y_test, hybrid.predict(test_df, X_test, vessel_classes))["R2"]

    assert hybrid_r2 > 0.9  # still a strong predictor
    assert abs(hybrid_r2 - physics_r2) < 0.05  # within a small band of physics alone, not dramatically better or worse


def test_residual_learnability_is_near_zero_or_negative_on_synthetic_data(vessel_catalog_path):
    """The core diagnostic behind the "don't expect hybrid to help here"
    finding: cross-validated R^2 of predicting the physics residual from
    features should be small or negative, confirming there is no systematic,
    learnable bias for a residual model to correct on this dataset."""
    vessel_classes, train_df, _, encoder = _prepare(vessel_catalog_path)
    X_train = build_feature_matrix(train_df, encoder)

    r2 = residual_learnability_r2(train_df, X_train, vessel_classes)
    assert r2 < 0.1  # near-zero or negative -- essentially unpredictable from these features
