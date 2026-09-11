"""Shared I/O helpers: YAML config loading and dataframe read/write."""

from pathlib import Path

import pandas as pd
import yaml


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_dataframe(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def load_dataframe(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)
