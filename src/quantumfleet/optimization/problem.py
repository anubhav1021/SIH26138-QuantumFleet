"""Mathematical formulation of the green-fleet deployment optimization problem.

**Decision variables** (see `optimization.encoding.Assignment`): for each
route, up to `N_SLOTS_PER_ROUTE` independent (vessel_class, fuel_type,
speed, use_shore_power, count) assignments -- a route can be served by a mix
of vessel/fuel combinations, not just one.

**Objectives** (minimize all three; computed in `optimization.qea.evaluate`):
  1. total fuel mass consumed (tonnes) over the planning period
  2. total lifecycle (well-to-wake) CO2e emissions (tonnes)
  3. total cost (USD): fuel/energy cost + vessel charter cost + any carbon levy

**Constraints** (see `optimization.constraints`):
  - cargo demand: each route's deliverable capacity must cover its
    cargo_demand_tonnes over the planning period
  - schedule reliability: each deployed assignment's one-way transit time
    must not exceed the route's max_transit_days
  - emission cap: each route's total CO2e must not exceed
    emission_cap_tonnes_co2e
  - fuel/vessel compatibility: fuel_type must be in both the route's
    allowed_fuel_types AND the vessel_class's compatible_fuels (enforced by
    repair, see `optimization.repair`)
  - shore power: use_shore_power only valid if the route's
    shore_power_available is True (enforced by repair)
"""

from dataclasses import dataclass

from quantumfleet.prediction.physics_model import VesselClass, load_vessel_catalog
from quantumfleet.utils.io import load_yaml

N_SLOTS_PER_ROUTE = 3


@dataclass(frozen=True)
class Route:
    route_id: str
    distance_nm: float
    cargo_demand_tonnes: float
    period_days: float
    max_transit_days: float
    allowed_fuel_types: tuple[str, ...]
    shore_power_available: bool
    emission_cap_tonnes_co2e: float
    # Optional, purely cosmetic -- never read by the optimizer, constraints, or
    # QEA. Only used by reporting/dashboard map visualization when present.
    # Every existing scenario YAML omits these and keeps working unchanged.
    origin_port: str | None = None
    destination_port: str | None = None
    origin_lat: float | None = None
    origin_lon: float | None = None
    destination_lat: float | None = None
    destination_lon: float | None = None


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    routes: tuple[Route, ...]
    carbon_price_usd_per_tonne: float = 0.0
    green_fuel_pathway: bool = False
    description: str = ""  # cosmetic only, never read by the optimizer


@dataclass(frozen=True)
class ProblemSpec:
    scenario: ScenarioConfig
    vessel_classes: dict[str, VesselClass]
    speed_bins_knots: tuple[float, ...]
    n_slots_per_route: int = N_SLOTS_PER_ROUTE


def _optional_float(r: dict, key: str) -> float | None:
    value = r.get(key)
    return float(value) if value is not None else None


def parse_route(r: dict) -> Route:
    """Builds one Route from a raw dict (one YAML list entry, or a dict built
    by the Scenario Builder UI). The single place that knows how a raw route
    dict becomes a `Route` -- used by both file-based loading and the
    in-memory scenario builder, so both paths produce identical objects."""
    return Route(
        route_id=r["route_id"],
        distance_nm=float(r["distance_nm"]),
        cargo_demand_tonnes=float(r["cargo_demand_tonnes"]),
        period_days=float(r["period_days"]),
        max_transit_days=float(r["max_transit_days"]),
        allowed_fuel_types=tuple(r["allowed_fuel_types"]),
        shore_power_available=bool(r["shore_power_available"]),
        emission_cap_tonnes_co2e=float(r["emission_cap_tonnes_co2e"]),
        origin_port=r.get("origin_port"),
        destination_port=r.get("destination_port"),
        origin_lat=_optional_float(r, "origin_lat"),
        origin_lon=_optional_float(r, "origin_lon"),
        destination_lat=_optional_float(r, "destination_lat"),
        destination_lon=_optional_float(r, "destination_lon"),
    )


def parse_scenario(raw: dict) -> ScenarioConfig:
    """Builds a ScenarioConfig from a raw dict (parsed YAML, or a dict
    assembled by the Scenario Builder UI before it's ever written to disk).
    This is the single conversion point `load_scenario` (files) and the
    scenario repository/builder (UI) both funnel through, so a UI-built
    scenario is indistinguishable from a file-loaded one by the time it
    reaches the optimizer."""
    routes = tuple(parse_route(r) for r in raw["routes"])
    return ScenarioConfig(
        name=raw.get("name", "scenario"),
        routes=routes,
        carbon_price_usd_per_tonne=float(raw.get("carbon_price_usd_per_tonne", 0.0)),
        green_fuel_pathway=bool(raw.get("green_fuel_pathway", False)),
        description=raw.get("description", ""),
    )


def load_scenario(path: str) -> ScenarioConfig:
    return parse_scenario(load_yaml(path))


def load_problem(scenario_path: str, vessel_catalog_path: str) -> ProblemSpec:
    scenario = load_scenario(scenario_path)
    vessel_classes, speed_bins = load_vessel_catalog(vessel_catalog_path)
    return ProblemSpec(scenario=scenario, vessel_classes=vessel_classes, speed_bins_knots=tuple(speed_bins))


def build_problem(scenario: ScenarioConfig, vessel_classes: dict[str, VesselClass], speed_bins_knots: list[float]) -> ProblemSpec:
    """Assembles a ProblemSpec directly from already-in-memory objects (no
    file I/O) -- what the Scenario Builder UI uses to hand a just-created or
    just-edited scenario to the SAME `QuantumEvolutionaryOptimizer` that
    file-based scenarios use. There is no second optimizer for custom
    scenarios; this function is the only difference, and it's just
    `load_problem` without the file reads."""
    return ProblemSpec(scenario=scenario, vessel_classes=vessel_classes, speed_bins_knots=tuple(speed_bins_knots))
