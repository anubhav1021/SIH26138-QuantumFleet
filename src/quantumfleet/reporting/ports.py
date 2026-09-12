"""A small curated lookup of major world port coordinates, used to resolve a
route's `origin_port`/`destination_port` NAME into map coordinates without
requiring a user to enter latitude/longitude by hand.

Deliberately not exhaustive and not a geocoding service call: an external
lookup would be a fragile dependency for a secondary visualization feature.
A port not in this list simply doesn't render on the map -- a graceful,
honest degradation (see `charts.route_map`), not a broken lookup. A route
can also set `origin_lat`/`origin_lon`/`destination_lat`/`destination_lon`
directly on the Route if precise coordinates matter more than convenience.
"""

MAJOR_PORTS: dict[str, tuple[float, float]] = {
    "shanghai": (31.2, 121.5),
    "singapore": (1.29, 103.85),
    "rotterdam": (51.9, 4.5),
    "ningbo": (29.87, 121.55),
    "shenzhen": (22.5, 114.1),
    "guangzhou": (23.1, 113.3),
    "busan": (35.1, 129.04),
    "qingdao": (36.07, 120.38),
    "hong kong": (22.3, 114.17),
    "los angeles": (33.73, -118.26),
    "long beach": (33.75, -118.2),
    "antwerp": (51.29, 4.4),
    "hamburg": (53.55, 9.99),
    "dubai": (25.27, 55.3),
    "jebel ali": (25.0, 55.06),
    "tianjin": (39.0, 117.7),
    "port klang": (3.0, 101.4),
    "new york": (40.67, -74.03),
    "kaohsiung": (22.6, 120.28),
    "colombo": (6.95, 79.85),
    "mumbai": (18.95, 72.85),
    "jawaharlal nehru": (18.95, 72.95),
    "chennai": (13.1, 80.3),
    "kolkata": (22.55, 88.3),
    "santos": (-23.96, -46.3),
    "sydney": (-33.87, 151.21),
    "melbourne": (-37.84, 144.93),
    "tokyo": (35.65, 139.77),
    "yokohama": (35.45, 139.65),
    "piraeus": (37.94, 23.64),
    "gothenburg": (57.7, 11.9),
    "panama city": (8.98, -79.52),
    "suez": (29.97, 32.55),
    "genoa": (44.41, 8.93),
    "valencia": (39.44, -0.32),
    "santa marta": (11.24, -74.2),
}


def resolve_port_coordinates(port_name: str | None) -> tuple[float, float] | None:
    if not port_name:
        return None
    return MAJOR_PORTS.get(port_name.strip().lower())
