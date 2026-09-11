import pytest

from quantumfleet.prediction.physics_model import (
    VesselClass,
    VoyageConditions,
    admiralty_power_kw,
    auxiliary_fuel_consumption_at_berth,
    displacement_tonnes,
    predict_fuel_consumption_physics,
    resistance_multiplier,
)

MEDIUM_CONTAINER = VesselClass(
    id="CONTAINER_MEDIUM", type="Container Ship", tier="Medium",
    dwt_tonnes=50_000, admiralty_coefficient=450, day_rate_usd=12_000,
    min_speed_knots=14, max_speed_knots=22, aux_load_kw=800,
    compatible_fuels=("HFO", "MDO", "LNG", "METHANOL"),
)


def test_displacement_increases_with_load_factor():
    d_low = displacement_tonnes(MEDIUM_CONTAINER, 0.2)
    d_high = displacement_tonnes(MEDIUM_CONTAINER, 1.0)
    assert d_low > 0
    assert d_high > d_low


def test_power_increases_cubically_with_speed():
    disp = displacement_tonnes(MEDIUM_CONTAINER, 0.8)
    p_slow = admiralty_power_kw(disp, 10, MEDIUM_CONTAINER.admiralty_coefficient)
    p_fast = admiralty_power_kw(disp, 20, MEDIUM_CONTAINER.admiralty_coefficient)
    assert p_fast > p_slow
    assert p_fast / p_slow == pytest.approx(8.0, rel=0.01)


def test_resistance_multiplier_increases_with_weather_severity():
    assert resistance_multiplier(0, 0, 0) == 1.0
    assert resistance_multiplier(4, 20, 0) > 1.0


def test_fuel_consumption_monotonic_in_speed_load_and_weather():
    base = VoyageConditions(speed_knots=14, load_factor=0.8, distance_nm=300)
    faster = VoyageConditions(speed_knots=18, load_factor=0.8, distance_nm=300)
    heavier = VoyageConditions(speed_knots=14, load_factor=1.0, distance_nm=300)
    rougher = VoyageConditions(speed_knots=14, load_factor=0.8, distance_nm=300, wave_height_m=4, headwind_knots=15)

    base_pred = predict_fuel_consumption_physics(MEDIUM_CONTAINER, base, "HFO")
    assert predict_fuel_consumption_physics(MEDIUM_CONTAINER, faster, "HFO").fuel_tonnes > base_pred.fuel_tonnes
    assert predict_fuel_consumption_physics(MEDIUM_CONTAINER, heavier, "HFO").fuel_tonnes > base_pred.fuel_tonnes
    assert predict_fuel_consumption_physics(MEDIUM_CONTAINER, rougher, "HFO").fuel_tonnes > base_pred.fuel_tonnes


def test_fuel_consumption_order_of_magnitude_is_plausible():
    conditions = VoyageConditions(speed_knots=14, load_factor=0.8, distance_nm=14 * 24)
    pred = predict_fuel_consumption_physics(MEDIUM_CONTAINER, conditions, "HFO")
    daily_fuel_tonnes = pred.fuel_tonnes / (pred.duration_hours / 24)
    assert 15 < daily_fuel_tonnes < 100


def test_alternative_fuel_changes_mass_not_power():
    conditions = VoyageConditions(speed_knots=14, load_factor=0.8, distance_nm=300)
    hfo = predict_fuel_consumption_physics(MEDIUM_CONTAINER, conditions, "HFO")
    methanol = predict_fuel_consumption_physics(MEDIUM_CONTAINER, conditions, "METHANOL")
    assert hfo.power_kw == methanol.power_kw
    assert methanol.fuel_tonnes > hfo.fuel_tonnes


def test_shore_power_avoids_onboard_fuel_burn():
    with_shore = auxiliary_fuel_consumption_at_berth(MEDIUM_CONTAINER, 24, "HFO", use_shore_power=True)
    without_shore = auxiliary_fuel_consumption_at_berth(MEDIUM_CONTAINER, 24, "HFO", use_shore_power=False)
    assert with_shore.fuel_tonnes == 0.0
    assert with_shore.co2e_tonnes > 0.0
    assert without_shore.fuel_tonnes > 0.0
