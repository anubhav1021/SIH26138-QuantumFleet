"""Shortest navigable sea distance between two named ports.

Great-circle distance is not usable here. Ships cannot cross land, so a
straight line understates real voyages by 30-55% on exactly the
intercontinental routes this project models -- Rotterdam-Shanghai by more than
half, because the straight line runs overland across Eurasia instead of
through Suez. Route distance feeds the Admiralty power relation, so that error
would propagate into every fuel, emissions, and cost figure downstream.

`searoute` instead solves a shortest path over a marine network graph, which
reproduces published sea distances to within roughly 1-10% and selects the
correct canal or strait (Suez, Panama, Malacca) on its own. It bundles its own
geodata, so unlike a geocoding or distance API there is no network call at
runtime and no key to manage -- the same constraint that kept
`reporting.ports` a local table rather than a service lookup.
"""

from functools import lru_cache

from quantumfleet.reporting.ports import resolve_port_coordinates


@lru_cache(maxsize=512)
def _sea_distance_between(origin_lat: float, origin_lon: float, dest_lat: float, dest_lon: float) -> float:
    # Imported lazily: searoute costs ~0.8s to import (it loads its marine
    # network), which is worth paying on first use rather than on every
    # dashboard boot, since most sessions never author a port pair.
    import searoute

    # searoute takes (lon, lat) -- the opposite order from our port table.
    route = searoute.searoute((origin_lon, origin_lat), (dest_lon, dest_lat), units="naut")
    return float(route["properties"]["length"])


def sea_distance_nm(origin_port: str | None, destination_port: str | None) -> float | None:
    """Sea distance in nautical miles between two port NAMES.

    Returns None if either name is not in the known-port table, which is the
    caller's signal that there is nothing to auto-fill.
    """
    origin = resolve_port_coordinates(origin_port)
    destination = resolve_port_coordinates(destination_port)
    if origin is None or destination is None:
        return None
    return _sea_distance_between(origin[0], origin[1], destination[0], destination[1])
