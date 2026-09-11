from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def vessel_catalog_path() -> str:
    return str(PROJECT_ROOT / "configs" / "vessel_types.yaml")


@pytest.fixture
def default_scenario_path() -> str:
    return str(PROJECT_ROOT / "configs" / "default_scenario.yaml")
