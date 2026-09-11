"""Prediction accuracy metrics and a multi-model comparison table."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": mean_squared_error(y_true, y_pred) ** 0.5,
        "MAPE": mean_absolute_percentage_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def compare_models(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """`results` maps a model name to its `regression_metrics()` dict."""
    return pd.DataFrame(results).T
