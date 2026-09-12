import math

from quantumfleet.reporting.ports import resolve_port_coordinates
from quantumfleet.scenarios.sea_distance import sea_distance_nm

EARTH_RADIUS_NM = 3440.065


def great_circle_nm(port_a: str, port_b: str) -> float:
    """Straight-line distance, for contrast with the routed sea distance."""
    lat1, lon1 = map(math.radians, resolve_port_coordinates(port_a))
    lat2, lon2 = map(math.radians, resolve_port_coordinates(port_b))
    inner = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return EARTH_RADIUS_NM * 2 * math.asin(math.sqrt(inner))


def test_unknown_port_returns_none():
    assert sea_distance_nm("Shanghai", "Nowheresville") is None
    assert sea_distance_nm("Nowheresville", "Shanghai") is None


def test_country_names_are_not_treated_as_ports():
    # A country is ambiguous (which Indian port?), so it must not silently
    # resolve to some arbitrary port and produce a confident wrong distance.
    assert sea_distance_nm("India", "China") is None


def test_missing_port_returns_none():
    assert sea_distance_nm(None, "Shanghai") is None
    assert sea_distance_nm("", "Shanghai") is None


def test_known_pair_is_case_insensitive():
    assert sea_distance_nm("SHANGHAI", "rotterdam") == sea_distance_nm("Shanghai", "Rotterdam")


def test_distance_is_symmetric():
    there = sea_distance_nm("Singapore", "Rotterdam")
    back = sea_distance_nm("Rotterdam", "Singapore")
    assert math.isclose(there, back, rel_tol=0.02)


def test_rotterdam_shanghai_matches_published_suez_routing():
    # Published sea distance via Suez is ~10,500 nm. This is the headline case
    # for the feature: the straight line is ~4,800 nm because it runs overland
    # across Eurasia.
    assert 9_500 <= sea_distance_nm("Rotterdam", "Shanghai") <= 11_500


def test_los_angeles_new_york_routes_around_the_americas():
    # ~4,700 nm via Panama, versus ~2,100 nm straight across the continent.
    assert 4_000 <= sea_distance_nm("Los Angeles", "New York") <= 5_500


def test_routed_distance_exceeds_great_circle_where_land_intervenes():
    """The core guarantee: this is a sea route, not a straight line.

    Every one of these pairs has land between the endpoints, so a correct
    router must return a materially longer path than the great circle. If this
    fails, the implementation has regressed to straight-line distance.
    """
    for origin, destination in [("Rotterdam", "Shanghai"), ("Mumbai", "Shanghai"), ("Los Angeles", "New York")]:
        routed = sea_distance_nm(origin, destination)
        straight = great_circle_nm(origin, destination)
        assert routed > straight * 1.3, f"{origin}->{destination}: routed {routed:.0f} nm vs great circle {straight:.0f} nm"


def test_routed_distance_is_never_shorter_than_great_circle():
    # A path over the surface cannot beat the geodesic. Allow a small tolerance
    # for the router's network resolution near the endpoints.
    for origin, destination in [("Shanghai", "Busan"), ("Hamburg", "Rotterdam"), ("Dubai", "Mumbai")]:
        assert sea_distance_nm(origin, destination) >= great_circle_nm(origin, destination) * 0.9
