"""Plain pandas sanity checks on generated (or real, swapped-in) voyage records."""

import pandas as pd

from quantumfleet.data_generation.schema import VoyageRecordColumns as C


def validate_voyage_records(df: pd.DataFrame) -> list[str]:
    """Returns a list of problem descriptions; an empty list means the dataframe passed."""
    problems: list[str] = []

    missing_cols = [c for c in C.ALL if c not in df.columns]
    if missing_cols:
        problems.append(f"missing columns: {missing_cols}")
        return problems

    if df[C.ALL].isna().any().any():
        bad_cols = df[C.ALL].columns[df[C.ALL].isna().any()].tolist()
        problems.append(f"unexpected NaNs in columns: {bad_cols}")

    if (df[C.SPEED_KNOTS] <= 0).any():
        problems.append("non-positive speed_knots present")
    if not df[C.LOAD_FACTOR].between(0, 1.01).all():
        problems.append("load_factor outside [0,1]")
    if (df[C.FUEL_CONSUMED_TONNES] <= 0).any():
        problems.append("non-positive fuel_consumed_tonnes present")
    if (df[C.WAVE_HEIGHT_M] < 0).any():
        problems.append("negative wave_height_m present")

    return problems


def assert_valid_voyage_records(df: pd.DataFrame) -> None:
    problems = validate_voyage_records(df)
    assert not problems, f"voyage records failed validation: {problems}"
