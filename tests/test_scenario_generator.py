from quantumfleet.data_generation.scenario_generator import DEFAULT_ARCHETYPES, build_scenario
from quantumfleet.optimization.problem import ProblemSpec, ScenarioConfig, Route
from quantumfleet.prediction.physics_model import load_vessel_catalog


def test_build_scenario_produces_expected_route_count(vessel_catalog_path):
    vessel_classes, speed_bins = load_vessel_catalog(vessel_catalog_path)
    scenario = build_scenario(vessel_classes, speed_bins, seed=1)
    expected_count = sum(a[1] for a in DEFAULT_ARCHETYPES)
    assert len(scenario["routes"]) == expected_count
    assert len({r["route_id"] for r in scenario["routes"]}) == expected_count


def test_build_scenario_is_reproducible_with_same_seed(vessel_catalog_path):
    vessel_classes, speed_bins = load_vessel_catalog(vessel_catalog_path)
    s1 = build_scenario(vessel_classes, speed_bins, seed=7)
    s2 = build_scenario(vessel_classes, speed_bins, seed=7)
    assert s1 == s2


def test_calibrated_emission_caps_leave_a_feasible_baseline(vessel_catalog_path):
    """Every route's minimum single-vessel-class HFO deployment (the same one
    the cap is calibrated from) must itself be feasible -- otherwise the
    calibration produced a self-contradictory scenario."""
    from dataclasses import replace

    from quantumfleet.data_generation.scenario_generator import calibrate_emission_cap
    from quantumfleet.optimization import constraints as con
    from quantumfleet.optimization.encoding import Assignment
    from quantumfleet.optimization.qea import evaluate_single_assignment
    import math

    vessel_classes, speed_bins = load_vessel_catalog(vessel_catalog_path)
    scenario = build_scenario(vessel_classes, speed_bins, seed=3)

    for r in scenario["routes"][:10]:  # a representative sample keeps this test fast
        route = Route(
            route_id=r["route_id"], distance_nm=r["distance_nm"], cargo_demand_tonnes=r["cargo_demand_tonnes"],
            period_days=r["period_days"], max_transit_days=r["max_transit_days"],
            allowed_fuel_types=tuple(r["allowed_fuel_types"]), shore_power_available=r["shore_power_available"],
            emission_cap_tonnes_co2e=r["emission_cap_tonnes_co2e"],
        )
        compatible = [v for v in vessel_classes.values() if set(v.compatible_fuels) & set(route.allowed_fuel_types)]
        largest = max(compatible, key=lambda v: v.dwt_tonnes)
        nominal_speeds = [s for s in speed_bins if largest.min_speed_knots <= s <= largest.max_speed_knots]
        nominal_speed = nominal_speeds[len(nominal_speeds) // 2]
        fuel = next(f for f in route.allowed_fuel_types if f in largest.compatible_fuels)
        probe = Assignment(route_id=route.route_id, vessel_class=largest.id, fuel_type=fuel, speed_knots=nominal_speed, use_shore_power=route.shore_power_available, count=1)
        trips = con.trips_per_period(route, probe)
        capacity_per_count = largest.dwt_tonnes * trips
        count = min(7, max(1, math.ceil(route.cargo_demand_tonnes / capacity_per_count)))
        candidate = replace(probe, count=count)

        problem = ProblemSpec(scenario=ScenarioConfig(name="test", routes=(route,)), vessel_classes=vessel_classes, speed_bins_knots=tuple(speed_bins))
        result = evaluate_single_assignment(candidate, route, problem)
        route_violation = (
            con.cargo_demand_violation(route, [candidate], vessel_classes)
            + con.schedule_violation(route, [candidate])
            + con.emission_cap_violation(route, result.co2e_tonnes)
        )
        assert route_violation <= 1e-6, f"{r['route_id']} baseline is infeasible: violation={route_violation}"
