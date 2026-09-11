"""Physics-only baseline predictions, for comparison against the data-driven
ML models in prediction.ml_model."""

import numpy as np
import pandas as pd

from quantumfleet.data_generation.schema import VoyageRecordColumns as C
from quantumfleet.prediction.physics_model import VesselClass, VoyageConditions, predict_fuel_consumption_physics


def physics_baseline_predictions(df: pd.DataFrame, vessel_classes: dict[str, VesselClass]) -> np.ndarray:
    """Runs the deterministic physics formula row-by-row over a voyage-record
    dataframe. Reconstructs headwind from wind_dir_deg/heading_deg using the
    same projection the synthetic generator's weather model used, so this
    baseline sees the same information the generator did (a fair comparison,
    not one crippled by discarding available weather columns). Uses a nominal
    24-hour leg for distance, matching how the generator built the records."""
    preds = np.empty(len(df))
    for i, row in enumerate(df.itertuples()):
        vessel = vessel_classes[getattr(row, C.VESSEL_CLASS)]
        headwind_knots = getattr(row, C.WIND_SPEED_KNOTS) * np.cos(
            np.radians(getattr(row, C.WIND_DIR_DEG) - getattr(row, C.HEADING_DEG))
        )
        conditions = VoyageConditions(
            speed_knots=getattr(row, C.SPEED_KNOTS),
            load_factor=getattr(row, C.LOAD_FACTOR),
            distance_nm=getattr(row, C.SPEED_KNOTS) * 24.0,
            wave_height_m=getattr(row, C.WAVE_HEIGHT_M),
            headwind_knots=headwind_knots,
        )
        pred = predict_fuel_consumption_physics(vessel, conditions, getattr(row, C.FUEL_TYPE))
        preds[i] = pred.fuel_tonnes
    return preds
