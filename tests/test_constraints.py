from quantumfleet.optimization import constraints as con
from quantumfleet.optimization.encoding import Assignment
from quantumfleet.optimization.problem import Route

ROUTE = Route(
    route_id="R1",
    distance_nm=2400.0,
    cargo_demand_tonnes=50_000.0,
    period_days=30.0,
    max_transit_days=10.0,
    allowed_fuel_types=("HFO", "MDO"),
    shore_power_available=True,
    emission_cap_tonnes_co2e=10_000.0,
)

VESSEL_CLASSES = {}  # populated per-test where dwt matters


def _assignment(speed=14.0, count=1, fuel="HFO", shore_power=False):
    return Assignment(route_id="R1", vessel_class="TANKER_LARGE", fuel_type=fuel, speed_knots=speed, use_shore_power=shore_power, count=count)


def test_one_way_transit_days_scales_inversely_with_speed():
    slow = con.one_way_transit_days(2400, 10)
    fast = con.one_way_transit_days(2400, 20)
    assert slow == 2 * fast


def test_trips_per_period_below_one_when_round_trip_exceeds_period():
    a = _assignment(speed=1.0)  # absurdly slow -> round trip (203 days) far exceeds the 30-day period
    assert con.trips_per_period(ROUTE, a) < 1.0


def test_cargo_demand_violation_zero_when_capacity_sufficient():
    a = _assignment(speed=16, count=3)
    vessel_classes = {"TANKER_LARGE": _fake_vessel(dwt=200_000)}
    violation = con.cargo_demand_violation(ROUTE, [a], vessel_classes)
    assert violation == 0.0


def test_cargo_demand_violation_positive_when_undersupplied():
    a = _assignment(speed=16, count=1)
    vessel_classes = {"TANKER_LARGE": _fake_vessel(dwt=1_000)}  # tiny vessel, nowhere near enough
    violation = con.cargo_demand_violation(ROUTE, [a], vessel_classes)
    assert violation > 0.0


def test_schedule_violation_zero_within_limit():
    a = _assignment(speed=16)  # one-way = 2400/16/24 = 6.25 days < max_transit=10
    assert con.schedule_violation(ROUTE, [a]) == 0.0


def test_schedule_violation_positive_when_too_slow():
    a = _assignment(speed=4)  # one-way = 2400/4/24 = 25 days > max_transit=10
    assert con.schedule_violation(ROUTE, [a]) > 0.0


def test_emission_cap_violation_zero_within_cap():
    assert con.emission_cap_violation(ROUTE, 5_000.0) == 0.0


def test_emission_cap_violation_positive_over_cap():
    assert con.emission_cap_violation(ROUTE, 20_000.0) > 0.0


def _fake_vessel(dwt):
    from quantumfleet.prediction.physics_model import VesselClass

    return VesselClass(
        id="TANKER_LARGE", type="Tanker", tier="Large", dwt_tonnes=dwt,
        admiralty_coefficient=520, day_rate_usd=20_000,
        min_speed_knots=12, max_speed_knots=16, aux_load_kw=1500,
        compatible_fuels=("HFO", "MDO", "LNG", "METHANOL", "AMMONIA"),
    )
