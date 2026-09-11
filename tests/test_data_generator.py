import pandas as pd

from quantumfleet.data_generation.generator import generate_voyage_records
from quantumfleet.data_generation.schema import VoyageRecordColumns as C
from quantumfleet.data_generation.validate import validate_voyage_records
from quantumfleet.prediction.physics_model import load_vessel_catalog


def test_generate_small_dataset_passes_validation(vessel_catalog_path):
    vessel_classes, _ = load_vessel_catalog(vessel_catalog_path)
    df = generate_voyage_records(vessel_classes, n_vessels=20, legs_per_vessel=10, seed=1)
    assert len(df) == 200
    assert validate_voyage_records(df) == []


def test_fuel_type_respects_vessel_compatibility(vessel_catalog_path):
    vessel_classes, _ = load_vessel_catalog(vessel_catalog_path)
    df = generate_voyage_records(vessel_classes, n_vessels=30, legs_per_vessel=10, seed=2)
    for vessel_class_id, group in df.groupby(C.VESSEL_CLASS):
        allowed = set(vessel_classes[vessel_class_id].compatible_fuels)
        assert set(group[C.FUEL_TYPE].unique()) <= allowed


def test_generation_is_reproducible_with_same_seed(vessel_catalog_path):
    vessel_classes, _ = load_vessel_catalog(vessel_catalog_path)
    df1 = generate_voyage_records(vessel_classes, n_vessels=10, legs_per_vessel=5, seed=7)
    df2 = generate_voyage_records(vessel_classes, n_vessels=10, legs_per_vessel=5, seed=7)
    pd.testing.assert_frame_equal(df1, df2)


def test_summary_statistics_are_plausible(vessel_catalog_path):
    vessel_classes, _ = load_vessel_catalog(vessel_catalog_path)
    df = generate_voyage_records(vessel_classes, n_vessels=100, legs_per_vessel=15, seed=3)
    # a laden medium/large vessel cruising should plausibly burn single- to
    # low-triple-digit tonnes/day; catches gross unit/formula errors early.
    assert 1.0 < df[C.FUEL_CONSUMED_TONNES].median() < 200.0
    assert df[C.CO2E_EMITTED_TONNES].median() > 0
