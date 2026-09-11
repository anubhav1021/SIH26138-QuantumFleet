import numpy as np

from quantumfleet.data_generation.generator import generate_voyage_records
from quantumfleet.prediction.features import build_feature_matrix, build_target, make_encoder
from quantumfleet.prediction.physics_model import load_vessel_catalog
from quantumfleet.prediction.qpso_tuner import DEFAULT_RF_BOUNDS, HyperparamBounds, quantum_pso_tune, rf_cv_rmse_objective

SPHERE_BOUNDS = [HyperparamBounds("x", -10, 10, integer=False), HyperparamBounds("y", -10, 10, integer=False)]


def _sphere(params: dict) -> float:
    return params["x"] ** 2 + params["y"] ** 2


def test_qpso_best_ever_score_is_monotonically_non_increasing():
    result = quantum_pso_tune(_sphere, bounds=SPHERE_BOUNDS, n_particles=8, n_iterations=10, seed=1)
    history = np.array(result.history)
    assert np.all(np.diff(history) <= 1e-9)


def test_qpso_converges_close_to_known_minimum():
    result = quantum_pso_tune(_sphere, bounds=SPHERE_BOUNDS, n_particles=15, n_iterations=30, seed=2)
    assert result.best_score < 1.0


def test_qpso_tunes_random_forest_hyperparams_within_bounds(vessel_catalog_path):
    vessel_classes, _ = load_vessel_catalog(vessel_catalog_path)
    df = generate_voyage_records(vessel_classes, n_vessels=30, legs_per_vessel=10, seed=5)
    encoder = make_encoder(list(vessel_classes.keys()))
    X = build_feature_matrix(df, encoder)
    y = build_target(df)

    objective = rf_cv_rmse_objective(X, y, cv=2)
    result = quantum_pso_tune(objective, bounds=DEFAULT_RF_BOUNDS, n_particles=5, n_iterations=3, seed=3)

    for b in DEFAULT_RF_BOUNDS:
        assert b.low <= result.best_params[b.name] <= b.high
    assert np.isfinite(result.best_score)
