"""Raw voyage-record dataframe -> (X, y) matrices for the ML prediction model."""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import OneHotEncoder

from quantumfleet.data_generation.schema import VoyageRecordColumns as C
from quantumfleet.fuels.properties import PROPULSION_FUELS

CATEGORICAL_FEATURES = [C.VESSEL_CLASS, C.FUEL_TYPE]
NUMERIC_FEATURES = [C.SPEED_KNOTS, C.LOAD_FACTOR, C.WAVE_HEIGHT_M, C.WIND_SPEED_KNOTS]


def train_test_split_by_vessel(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Splits by vessel_id, not by row: each vessel carries a fixed efficiency
    offset (data_generation.vessel_profiles), so a row-level split would leak
    that offset across train/test and overstate accuracy."""
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(splitter.split(df, groups=df[C.GROUP_COLUMN]))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)


def make_encoder(vessel_class_ids: list[str]) -> OneHotEncoder:
    """A OneHotEncoder over a FIXED category universe (every known vessel
    class + all 6 propulsion fuels), not just whatever happens to appear in
    one dataframe. This means the identical encoder can be reconstructed
    anywhere -- training, inference, the optimizer's ML cross-check -- from
    just the vessel catalog, with no fitted object to save/load, and a
    random synthetic sample that happens to omit a rare category never
    produces a mismatched feature space downstream."""
    encoder = OneHotEncoder(categories=[list(vessel_class_ids), list(PROPULSION_FUELS)], handle_unknown="ignore", sparse_output=False)
    encoder.fit(pd.DataFrame({C.VESSEL_CLASS: [vessel_class_ids[0]], C.FUEL_TYPE: [PROPULSION_FUELS[0]]}))
    return encoder


def build_feature_matrix(df: pd.DataFrame, encoder: OneHotEncoder) -> np.ndarray:
    cat_matrix = encoder.transform(df[CATEGORICAL_FEATURES])
    numeric_matrix = df[NUMERIC_FEATURES].to_numpy(dtype=float)
    return np.hstack([numeric_matrix, cat_matrix])


def feature_names(encoder: OneHotEncoder) -> list[str]:
    """Column names for `build_feature_matrix`'s output, in the same
    numeric-then-categorical order -- for feature-importance plots, since
    the raw feature matrix is otherwise just an unlabeled array of columns."""
    return list(NUMERIC_FEATURES) + list(encoder.get_feature_names_out(CATEGORICAL_FEATURES))


def build_target(df: pd.DataFrame) -> np.ndarray:
    return df[C.TARGET_COLUMN].to_numpy(dtype=float)
