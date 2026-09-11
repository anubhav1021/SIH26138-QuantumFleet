"""Generates realistic multi-route scenarios (route distances/demands drawn
from archetypes) with emission caps CALIBRATED rather than guessed -- for
each route, find the largest DWT-compatible vessel class, size its count to
satisfy cargo demand at a nominal mid-range speed, compute the resulting
HFO-baseline CO2e via the platform's own physics model, and cap at
`emission_cap_multiplier` times that reference. This is the method used to
fix `configs/default_scenario.yaml`'s originally unrealistic caps (see
docs/algorithm_details.md) -- it keeps every route's cap binding (rules out
wasteful oversized deployments) while leaving a real feasible region.
"""

import math
from dataclasses import replace

import numpy as np

from quantumfleet.optimization import constraints as con
from quantumfleet.optimization.encoding import Assignment
from quantumfleet.optimization.problem import ProblemSpec, Route, ScenarioConfig
from quantumfleet.optimization.qea import evaluate_single_assignment
from quantumfleet.prediction.physics_model import VesselClass

# (name, count, distance_range_nm, demand_range_tonnes, schedule_buffer_days, allowed_fuels, shore_power_probability)
DEFAULT_ARCHETYPES = [
    ("LONGHAUL", 15, (4000, 11000), (60_000, 150_000), 10, ["HFO", "MDO", "LNG", "METHANOL", "AMMONIA"], 0.5),
    ("REGIONAL", 20, (1000, 4000), (20_000, 70_000), 6, ["HFO", "MDO", "LNG", "METHANOL"], 0.6),
    ("SHORTSEA", 15, (100, 1000), (5_000, 25_000), 3, ["HFO", "MDO", "LNG", "HYDROGEN"], 0.8),
]

NOMINAL_ROUND_TRIP_SPEED_KNOTS = 12.0  # sizes a generous schedule buffer only, not an optimizer input
EMISSION_CAP_MULTIPLIER = 1.3


def draw_routes(archetypes: list[tuple], seed: int) -> list[dict]:
    rng = np.random.default_rng(seed)
    routes, idx = [], 0
    for name, count, dist_range, demand_range, buffer_days, fuels, shore_prob in archetypes:
        for _ in range(count):
            distance = float(rng.uniform(*dist_range))
            baseline_transit_days = distance / NOMINAL_ROUND_TRIP_SPEED_KNOTS / 24.0
            routes.append(
                {
                    "route_id": f"{name}_{idx:03d}",
                    "distance_nm": round(distance),
                    "cargo_demand_tonnes": round(float(rng.uniform(*demand_range)), -2),
                    "period_days": 30.0,
                    "max_transit_days": round(baseline_transit_days * 1.3 + buffer_days, 1),
                    "allowed_fuel_types": list(fuels),  # a fresh list per route, not a shared reference
                    "shore_power_available": bool(rng.random() < shore_prob),
                }
            )
            idx += 1
    return routes


def calibrate_emission_cap(route_dict: dict, vessel_classes: dict[str, VesselClass], speed_bins: list[float]) -> float:
    route = Route(
        route_id=route_dict["route_id"], distance_nm=route_dict["distance_nm"], cargo_demand_tonnes=route_dict["cargo_demand_tonnes"],
        period_days=route_dict["period_days"], max_transit_days=route_dict["max_transit_days"],
        allowed_fuel_types=tuple(route_dict["allowed_fuel_types"]), shore_power_available=route_dict["shore_power_available"],
        emission_cap_tonnes_co2e=0.0,  # placeholder: unused by evaluate_single_assignment
    )
    compatible = [v for v in vessel_classes.values() if set(v.compatible_fuels) & set(route.allowed_fuel_types)]
    largest = max(compatible, key=lambda v: v.dwt_tonnes)
    nominal_speeds = [s for s in speed_bins if largest.min_speed_knots <= s <= largest.max_speed_knots]
    nominal_speed = nominal_speeds[len(nominal_speeds) // 2]
    fuel = next(f for f in route.allowed_fuel_types if f in largest.compatible_fuels)

    probe = Assignment(route_id=route.route_id, vessel_class=largest.id, fuel_type=fuel, speed_knots=nominal_speed, use_shore_power=route.shore_power_available, count=1)
    trips = con.trips_per_period(route, probe)
    capacity_per_count = largest.dwt_tonnes * trips if trips > 0 else 0.0
    count = min(7, max(1, math.ceil(route.cargo_demand_tonnes / capacity_per_count))) if capacity_per_count > 0 else 7
    candidate = replace(probe, count=count)

    single_route_problem = ProblemSpec(scenario=ScenarioConfig(name="calibration", routes=(route,)), vessel_classes=vessel_classes, speed_bins_knots=tuple(speed_bins))
    result = evaluate_single_assignment(candidate, route, single_route_problem)
    reference_co2e = result.co2e_tonnes if result is not None else route.cargo_demand_tonnes * 0.05
    return round(reference_co2e * EMISSION_CAP_MULTIPLIER, -1)


def build_scenario(
    vessel_classes: dict[str, VesselClass],
    speed_bins: list[float],
    seed: int,
    name: str = "generated_scenario",
    carbon_price_usd_per_tonne: float = 50.0,
    archetypes: list[tuple] = DEFAULT_ARCHETYPES,
) -> dict:
    routes = draw_routes(archetypes, seed)
    for r in routes:
        r["emission_cap_tonnes_co2e"] = calibrate_emission_cap(r, vessel_classes, speed_bins)
    return {"name": name, "carbon_price_usd_per_tonne": carbon_price_usd_per_tonne, "green_fuel_pathway": False, "routes": routes}
