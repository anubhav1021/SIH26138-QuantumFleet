"""Thin wrapper around scikit-learn regressors for fuel-consumption prediction.

Kept deliberately swappable between a plain linear baseline and tree-ensemble
models so `prediction.evaluate` can produce a fair physics-vs-linear-vs-ML
comparison table from one consistent interface.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

ModelKind = Literal["random_forest", "gradient_boosting", "linear"]


def _make_estimator(kind: ModelKind, **hyperparams):
    if kind == "random_forest":
        return RandomForestRegressor(random_state=42, **hyperparams)
    if kind == "gradient_boosting":
        return GradientBoostingRegressor(random_state=42, **hyperparams)
    if kind == "linear":
        return LinearRegression()
    raise ValueError(f"unknown model kind: {kind}")


@dataclass
class FuelPredictionModel:
    kind: ModelKind = "random_forest"
    hyperparams: dict = field(default_factory=dict)
    estimator: object = field(init=False, default=None)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FuelPredictionModel":
        self.estimator = _make_estimator(self.kind, **self.hyperparams)
        self.estimator.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.estimator is None:
            raise RuntimeError("model not fitted yet")
        return self.estimator.predict(X)

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "FuelPredictionModel":
        return joblib.load(path)
