"""Repair operators enforcing hard feasibility rules that would be wasteful
for evolution to discover by trial and error -- an incompatible fuel/vessel
pairing, an out-of-range speed, or shore power at a berth without a hookup
are never useful, so they're corrected deterministically rather than merely
penalized and left for the search to stumble onto a fix.
"""

from dataclasses import replace

from quantumfleet.optimization.encoding import Assignment, FleetPlan
from quantumfleet.optimization.problem import ProblemSpec


def fix_fuel_compatibility(assignment: Assignment, route_allowed_fuels: tuple[str, ...], vessel_compatible_fuels: tuple[str, ...]) -> Assignment:
    allowed = [f for f in route_allowed_fuels if f in vessel_compatible_fuels]
    if not allowed:
        # route and vessel share no allowed fuel: fall back to the vessel's
        # own compatible list so the assignment is at least well-formed;
        # constraint-dominance penalizes this route/vessel pairing downstream.
        allowed = list(vessel_compatible_fuels)
    if assignment.fuel_type not in allowed:
        return replace(assignment, fuel_type=allowed[0])
    return assignment


def clip_speed_to_class_range(assignment: Assignment, min_speed: float, max_speed: float, speed_bins: tuple[float, ...]) -> Assignment:
    if min_speed <= assignment.speed_knots <= max_speed:
        return assignment
    candidates = [s for s in speed_bins if min_speed <= s <= max_speed]
    if not candidates:
        return assignment
    nearest = min(candidates, key=lambda s: abs(s - assignment.speed_knots))
    return replace(assignment, speed_knots=nearest)


def force_shore_power_to_available_routes(assignment: Assignment, shore_power_available: bool) -> Assignment:
    if assignment.use_shore_power and not shore_power_available:
        return replace(assignment, use_shore_power=False)
    return assignment


def repair_plan(plan: FleetPlan, problem: ProblemSpec) -> FleetPlan:
    routes_by_id = {r.route_id: r for r in problem.scenario.routes}
    repaired = []
    for a in plan.assignments:
        route = routes_by_id[a.route_id]
        vessel = problem.vessel_classes[a.vessel_class]

        a = fix_fuel_compatibility(a, route.allowed_fuel_types, vessel.compatible_fuels)
        a = clip_speed_to_class_range(a, vessel.min_speed_knots, vessel.max_speed_knots, problem.speed_bins_knots)
        a = force_shore_power_to_available_routes(a, route.shore_power_available)
        repaired.append(a)
    return FleetPlan(assignments=tuple(repaired))
