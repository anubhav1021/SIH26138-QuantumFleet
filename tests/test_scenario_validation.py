from quantumfleet.optimization.problem import load_problem
from quantumfleet.scenarios.validation import (
    has_errors,
    max_feasible_capacity_tonnes,
    validate_route_dict,
    validate_scenario_dict,
    validate_scenario_feasibility,
    validate_vessel_dict,
    validate_vessel_speed_coverage,
)

VALID_ROUTE = {
    "route_id": "R1",
    "distance_nm": 1000,
    "cargo_demand_tonnes": 5000,
    "period_days": 30,
    "max_transit_days": 10,
    "allowed_fuel_types": ["HFO", "MDO"],
    "shore_power_available": True,
    "emission_cap_tonnes_co2e": 5000,
}

VALID_VESSEL = {
    "id": "TEST_VESSEL",
    "type": "Container Ship",
    "tier": "Medium",
    "dwt_tonnes": 40000,
    "admiralty_coefficient": 440,
    "day_rate_usd": 10000,
    "min_speed_knots": 12,
    "max_speed_knots": 20,
    "aux_load_kw": 700,
    "compatible_fuels": ["HFO", "MDO", "LNG"],
}


def test_valid_route_has_no_errors():
    assert not has_errors(validate_route_dict(VALID_ROUTE))


def test_missing_required_route_field_is_an_error():
    issues = validate_route_dict({**VALID_ROUTE, "distance_nm": None})
    assert has_errors(issues)
    assert any(i.field == "distance_nm" for i in issues)


def test_negative_distance_is_an_error():
    issues = validate_route_dict({**VALID_ROUTE, "distance_nm": -5})
    assert has_errors(issues)


def test_duplicate_route_id_is_an_error():
    issues = validate_route_dict(VALID_ROUTE, existing_route_ids={"R1"})
    assert has_errors(issues)
    assert any("unique" in i.message for i in issues)


def test_unknown_fuel_type_is_an_error():
    issues = validate_route_dict({**VALID_ROUTE, "allowed_fuel_types": ["UNOBTANIUM"]})
    assert has_errors(issues)


def test_empty_allowed_fuels_is_an_error():
    issues = validate_route_dict({**VALID_ROUTE, "allowed_fuel_types": []})
    assert has_errors(issues)


def test_low_emission_cap_is_a_warning_not_an_error():
    issues = validate_route_dict({**VALID_ROUTE, "emission_cap_tonnes_co2e": 0})
    assert not has_errors(issues)
    assert issues and issues[0].severity == "warning"


def test_valid_vessel_has_no_errors():
    assert not has_errors(validate_vessel_dict(VALID_VESSEL))


def test_min_speed_above_max_speed_is_an_error():
    issues = validate_vessel_dict({**VALID_VESSEL, "min_speed_knots": 25, "max_speed_knots": 20})
    assert has_errors(issues)


def test_duplicate_vessel_id_is_an_error():
    issues = validate_vessel_dict(VALID_VESSEL, existing_ids={"TEST_VESSEL"})
    assert has_errors(issues)


def test_negative_charter_cost_is_an_error():
    issues = validate_vessel_dict({**VALID_VESSEL, "day_rate_usd": -100})
    assert has_errors(issues)


def test_vessel_speed_range_not_covering_any_bin_is_flagged():
    issues = validate_vessel_speed_coverage({**VALID_VESSEL, "min_speed_knots": 22.5, "max_speed_knots": 23.5}, speed_bins_knots=[10, 12, 14, 16, 18, 20, 22])
    assert has_errors(issues)


def test_vessel_speed_range_covering_a_bin_is_not_flagged():
    issues = validate_vessel_speed_coverage(VALID_VESSEL, speed_bins_knots=[10, 12, 14, 16, 18, 20, 22])
    assert not issues


def test_valid_scenario_dict_has_no_errors():
    scenario = {"name": "Test", "routes": [VALID_ROUTE], "carbon_price_usd_per_tonne": 10}
    assert not has_errors(validate_scenario_dict(scenario))


def test_scenario_without_name_is_an_error():
    issues = validate_scenario_dict({"name": "", "routes": [VALID_ROUTE]})
    assert has_errors(issues)


def test_scenario_without_routes_is_an_error():
    issues = validate_scenario_dict({"name": "Test", "routes": []})
    assert has_errors(issues)


def test_scenario_duplicate_route_ids_across_list_is_an_error():
    issues = validate_scenario_dict({"name": "Test", "routes": [VALID_ROUTE, VALID_ROUTE]})
    assert has_errors(issues)


def test_max_feasible_capacity_is_positive_for_default_scenario(vessel_catalog_path, default_scenario_path):
    problem = load_problem(default_scenario_path, vessel_catalog_path)
    route = problem.scenario.routes[0]
    ceiling = max_feasible_capacity_tonnes(route, problem.vessel_classes, list(problem.speed_bins_knots))
    assert ceiling > 0


def test_max_feasible_capacity_zero_when_no_vessel_compatible(vessel_catalog_path, default_scenario_path):
    from dataclasses import replace

    problem = load_problem(default_scenario_path, vessel_catalog_path)
    impossible_route = replace(problem.scenario.routes[0], allowed_fuel_types=("UNOBTANIUM",))
    ceiling = max_feasible_capacity_tonnes(impossible_route, problem.vessel_classes, list(problem.speed_bins_knots))
    assert ceiling == 0.0


def test_calibrated_default_scenario_passes_feasibility_prechecks(vessel_catalog_path, default_scenario_path):
    """The shipped default_scenario.yaml was specifically calibrated to be
    feasible (see docs/algorithm_details.md) -- the pre-check must agree."""
    problem = load_problem(default_scenario_path, vessel_catalog_path)
    issues = validate_scenario_feasibility(problem.scenario, problem.vessel_classes, list(problem.speed_bins_knots))
    assert not has_errors(issues)


def test_absurd_cargo_demand_triggers_feasibility_warning(vessel_catalog_path, default_scenario_path):
    from dataclasses import replace

    problem = load_problem(default_scenario_path, vessel_catalog_path)
    absurd_scenario = replace(
        problem.scenario,
        routes=(replace(problem.scenario.routes[0], cargo_demand_tonnes=10_000_000_000.0),),
    )
    issues = validate_scenario_feasibility(absurd_scenario, problem.vessel_classes, list(problem.speed_bins_knots))
    assert any("exceeds the maximum" in i.message for i in issues)
