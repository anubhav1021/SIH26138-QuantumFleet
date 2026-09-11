"""Constraint checks: cargo demand, schedule reliability, and emission caps.

Each returns a normalized violation score (0.0 = fully satisfied); summed
per route by `optimization.qea.evaluate_by_route` into the total used by
`optimization.pareto.dominates` for constraint-dominance (Deb's rule):
feasible plans always beat infeasible ones, and among infeasible plans,
lower total violation wins.
"""

from quantumfleet.constants import DEFAULT_BERTH_TIME_DAYS
from quantumfleet.optimization.encoding import Assignment
from quantumfleet.optimization.problem import Route
from quantumfleet.prediction.physics_model import VesselClass


def one_way_transit_days(distance_nm: float, speed_knots: float) -> float:
    if speed_knots <= 0:
        return float("inf")
    return (distance_nm / speed_knots) / 24.0


def round_trip_days(route: Route, assignment: Assignment) -> float:
    return 2 * one_way_transit_days(route.distance_nm, assignment.speed_knots) + 2 * DEFAULT_BERTH_TIME_DAYS


def trips_per_period(route: Route, assignment: Assignment) -> float:
    rt = round_trip_days(route, assignment)
    if rt <= 0 or rt == float("inf"):
        return 0.0
    return route.period_days / rt


def route_capacity_tonnes(route: Route, assignments: list[Assignment], vessel_classes: dict[str, VesselClass]) -> float:
    return sum(
        vessel_classes[a.vessel_class].dwt_tonnes * a.count * trips_per_period(route, a)
        for a in assignments
        if a.count > 0
    )


def cargo_demand_violation(route: Route, assignments: list[Assignment], vessel_classes: dict[str, VesselClass]) -> float:
    if route.cargo_demand_tonnes <= 0:
        return 0.0
    capacity = route_capacity_tonnes(route, assignments, vessel_classes)
    shortfall = max(0.0, route.cargo_demand_tonnes - capacity)
    return shortfall / route.cargo_demand_tonnes


def schedule_violation(route: Route, assignments: list[Assignment]) -> float:
    """Fraction by which the slowest deployed assignment's one-way transit
    time exceeds max_transit_days; 0 if all deployed assignments (or none)
    are within schedule."""
    active = [a for a in assignments if a.count > 0]
    if not active:
        return 0.0
    worst = max(one_way_transit_days(route.distance_nm, a.speed_knots) for a in active)
    if worst <= route.max_transit_days:
        return 0.0
    return (worst - route.max_transit_days) / route.max_transit_days


def emission_cap_violation(route: Route, route_co2e_tonnes: float) -> float:
    if route.emission_cap_tonnes_co2e <= 0:
        return 0.0
    excess = max(0.0, route_co2e_tonnes - route.emission_cap_tonnes_co2e)
    return excess / route.emission_cap_tonnes_co2e
