from quantumfleet.optimization.problem import Route
from quantumfleet.reporting.charts import route_map
from quantumfleet.reporting.ports import resolve_port_coordinates

BASE_ROUTE_KWARGS = dict(distance_nm=1000.0, cargo_demand_tonnes=1000.0, period_days=30.0, max_transit_days=10.0, allowed_fuel_types=("HFO",), shore_power_available=False, emission_cap_tonnes_co2e=1000.0)


def test_resolve_port_coordinates_known_port_case_insensitive():
    assert resolve_port_coordinates("Shanghai") is not None
    assert resolve_port_coordinates("SHANGHAI") == resolve_port_coordinates("shanghai")


def test_resolve_port_coordinates_unknown_port_returns_none():
    assert resolve_port_coordinates("Nowheresville") is None


def test_resolve_port_coordinates_none_input():
    assert resolve_port_coordinates(None) is None
    assert resolve_port_coordinates("") is None


def test_route_map_none_when_no_route_has_coordinates():
    routes = [Route(route_id="R1", **BASE_ROUTE_KWARGS)]
    assert route_map(routes) is None


def test_route_map_renders_when_a_route_has_recognized_ports():
    routes = [Route(route_id="R1", origin_port="Shanghai", destination_port="Rotterdam", **BASE_ROUTE_KWARGS)]
    fig = route_map(routes)
    assert fig is not None
    assert len(fig.data) == 1


def test_route_map_uses_explicit_coordinates_over_port_name_lookup_failure():
    routes = [Route(route_id="R1", origin_lat=10.0, origin_lon=20.0, destination_lat=30.0, destination_lon=40.0, **BASE_ROUTE_KWARGS)]
    fig = route_map(routes)
    assert fig is not None


def test_route_map_skips_routes_with_only_one_endpoint_resolvable():
    routes = [Route(route_id="R1", origin_port="Shanghai", destination_port="Nowheresville", **BASE_ROUTE_KWARGS)]
    assert route_map(routes) is None


def test_route_map_mixed_routes_only_plots_the_resolvable_one():
    routes = [
        Route(route_id="R1", origin_port="Shanghai", destination_port="Rotterdam", **BASE_ROUTE_KWARGS),
        Route(route_id="R2", **BASE_ROUTE_KWARGS),  # no ports set
    ]
    fig = route_map(routes)
    assert fig is not None
    assert len(fig.data) == 1
