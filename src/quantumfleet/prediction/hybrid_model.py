"""Physics-informed hybrid fuel prediction: a residual-correction ML model
layered on top of the physics baseline (predicts `actual - physics`, added
back to the physics estimate) rather than replacing physics outright.

**Honest empirical finding** (see `docs/algorithm_details.md` for the full
investigation): on this platform's own synthetic dataset, the residual
carries almost no feature-predictable signal -- cross-validated R^2 of
predicting the residual from features is *negative* (worse than predicting
the mean residual). That's because the dataset's only error sources are a
per-vessel random efficiency offset (uncorrelated with any feature the
model observes, by construction -- see `data_generation.vessel_profiles`)
and i.i.d. log-normal measurement noise -- neither is a systematic,
feature-correlated bias for a residual model to learn. So on THIS data, the
hybrid model performs on par with, not better than, physics alone.

The architecture is still built and tested properly because on real voyage
data, physics-model residuals are typically NOT pure noise: hull fouling,
engine degradation, and other effects the Admiralty formula doesn't capture
usually correlate with observable features (vessel age, route, season),
which is exactly the situation this architecture is designed for.

**Why this is not used inside the QEA fitness function**: given the finding
above, there is no accuracy benefit to justify the extra cost (a dataframe
+ trained-model predict call per assignment, versus a closed-form physics
formula) in the optimizer's hot inner loop. QEA continues to use the
physics formula directly, exactly as before -- see `optimization.qea`.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

from quantumfleet.benchmarking.baselines_prediction import physics_baseline_predictions
from quantumfleet.data_generation.schema import VoyageRecordColumns as C
from quantumfleet.prediction.physics_model import VesselClass


@dataclass
class HybridFuelPredictionModel:
    n_estimators: int = 200
    residual_model: RandomForestRegressor | None = field(init=False, default=None)

    def fit(self, df: pd.DataFrame, X: np.ndarray, vessel_classes: dict[str, VesselClass]) -> "HybridFuelPredictionModel":
        physics_pred = physics_baseline_predictions(df, vessel_classes)
        residual = df[C.TARGET_COLUMN].to_numpy(dtype=float) - physics_pred
        self.residual_model = RandomForestRegressor(random_state=42, n_estimators=self.n_estimators).fit(X, residual)
        return self

    def predict(self, df: pd.DataFrame, X: np.ndarray, vessel_classes: dict[str, VesselClass]) -> np.ndarray:
        if self.residual_model is None:
            raise RuntimeError("model not fitted yet")
        physics_pred = physics_baseline_predictions(df, vessel_classes)
        return physics_pred + self.residual_model.predict(X)


def residual_learnability_r2(df: pd.DataFrame, X: np.ndarray, vessel_classes: dict[str, VesselClass], cv: int = 3, seed: int = 42) -> float:
    """Cross-validated R^2 of predicting the physics residual from features
    alone. Close to 0 (or negative) means the residual is essentially
    unpredictable noise given these features -- a hybrid model won't help.
    Meaningfully positive means there's a systematic, learnable physics-model
    bias worth correcting -- exactly the situation the hybrid model targets."""
    physics_pred = physics_baseline_predictions(df, vessel_classes)
    residual = df[C.TARGET_COLUMN].to_numpy(dtype=float) - physics_pred
    model = RandomForestRegressor(random_state=seed, n_estimators=100)
    scores = cross_val_score(model, X, residual, cv=cv, scoring="r2")
    return float(scores.mean())
