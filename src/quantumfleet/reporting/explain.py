"""Per-route constraint-satisfaction breakdown, plain-language plan
explanations, baseline comparison, and Pareto-archive selection helpers.

Shared by the dashboard's "Recommended Fleet Plan", "Why This Plan", and
"Baseline vs. Optimized" sections and the exported report, so every number
shown anywhere is computed once, here, from the actual constraint and
objective functions -- never restated or approximated by hand in the UI.
"""

from dataclasses import dataclass

from quantumfleet.optimization import constraints as con
from quantumfleet.optimization.encoding import FleetPlan
from quantumfleet.optimization.pareto import ArchiveEntry, Objectives, ParetoArchive
from quantumfleet.optimization.problem import ProblemSpec, Route
from quantumfleet.optimization.qea import evaluate_single_assignment


@dataclass(frozen=True)
class RouteFeasibility:
    route_id: str
    cargo_demand_tonnes: float
    cargo_capacity_tonnes: float
    cargo_fulfillment_pct: float
    schedule_ok: bool
    worst_transit_days: float
    max_transit_days: float
    emission_cap_ok: bool
    route_co2e_tonnes: float
    emission_cap_tonnes_co2e: float
    fuel_types_used: tuple[str, ...]
    has_any_assignment: bool


def explain_route_feasibility(route: Route, plan: FleetPlan, problem: ProblemSpec) -> RouteFeasibility:
    assignments = [a for a in plan.assignments if a.route_id == route.route_id and a.count > 0]
    capacity = con.route_capacity_tonnes(route, assignments, problem.vessel_classes)
    fulfillment_pct = min(100.0, 100.0 * capacity / route.cargo_demand_tonnes) if route.cargo_demand_tonnes > 0 else 100.0

    worst_transit = max((con.one_way_transit_days(route.distance_nm, a.speed_knots) for a in assignments), default=0.0)
    schedule_ok = worst_transit <= route.max_transit_days

    route_co2e = sum((evaluate_single_assignment(a, route, problem) or Objectives(0, 0, 0, 0)).co2e_tonnes for a in assignments)
    emission_cap_ok = route_co2e <= route.emission_cap_tonnes_co2e

    return RouteFeasibility(
        route_id=route.route_id,
        cargo_demand_tonnes=route.cargo_demand_tonnes,
        cargo_capacity_tonnes=capacity,
        cargo_fulfillment_pct=fulfillment_pct,
        schedule_ok=schedule_ok,
        worst_transit_days=worst_transit,
        max_transit_days=route.max_transit_days,
        emission_cap_ok=emission_cap_ok,
        route_co2e_tonnes=route_co2e,
        emission_cap_tonnes_co2e=route.emission_cap_tonnes_co2e,
        fuel_types_used=tuple(sorted({a.fuel_type for a in assignments})),
        has_any_assignment=bool(assignments),
    )


def explain_plan_feasibility(plan: FleetPlan, problem: ProblemSpec) -> list[RouteFeasibility]:
    return [explain_route_feasibility(route, plan, problem) for route in problem.scenario.routes]


def plan_summary_metrics(plan: FleetPlan, objectives: Objectives, problem: ProblemSpec) -> dict:
    """Aggregate, human-facing summary for the Recommended Fleet Plan
    section: fulfillment/compliance rates, carbon cost split out from total
    cost, vessels used, fuel mix. Carbon cost is recoverable exactly (not
    estimated) because `qea.evaluate_by_route` adds it as one separate term
    (`carbon_price_usd_per_tonne * co2e_tonnes`) on top of the per-assignment
    fuel/charter costs -- so subtracting it back out is exact, not a guess."""
    route_feasibilities = explain_plan_feasibility(plan, problem)
    n_routes = len(route_feasibilities)

    active = [a for a in plan.assignments if a.count > 0]
    fuel_mix: dict[str, int] = {}
    for a in active:
        fuel_mix[a.fuel_type] = fuel_mix.get(a.fuel_type, 0) + a.count

    carbon_cost_usd = problem.scenario.carbon_price_usd_per_tonne * objectives.co2e_tonnes

    return {
        "cargo_fulfillment_pct": (sum(r.cargo_fulfillment_pct for r in route_feasibilities) / n_routes) if n_routes else 0.0,
        "schedule_compliant_routes": sum(1 for r in route_feasibilities if r.schedule_ok),
        "emission_compliant_routes": sum(1 for r in route_feasibilities if r.emission_cap_ok),
        "n_routes": n_routes,
        "vessels_used": sum(a.count for a in active),
        "fuel_mix": fuel_mix,
        "carbon_cost_usd": carbon_cost_usd,
        "operational_cost_usd": objectives.cost_usd - carbon_cost_usd,
        "route_feasibilities": route_feasibilities,
    }


def explain_why_plan_chosen(objectives: Objectives, route_feasibilities: list[RouteFeasibility], archive: ParetoArchive, selected: ArchiveEntry) -> list[str]:
    """Plain-language bullets, each derived directly from a measured
    value -- no fabricated reasoning about *why* the QEA chose particular
    vessels/speeds/fuels beyond what the numbers themselves show."""
    n = len(route_feasibilities)
    lines = []

    fulfilled = sum(1 for r in route_feasibilities if r.cargo_fulfillment_pct >= 99.9)
    lines.append(f"{'✓' if fulfilled == n else '⚠'} Cargo demand fully met on {fulfilled}/{n} routes.")

    on_schedule = sum(1 for r in route_feasibilities if r.schedule_ok)
    lines.append(f"{'✓' if on_schedule == n else '⚠'} Schedule (max transit time) met on {on_schedule}/{n} routes.")

    within_cap = sum(1 for r in route_feasibilities if r.emission_cap_ok)
    lines.append(f"{'✓' if within_cap == n else '⚠'} Emission cap respected on {within_cap}/{n} routes.")

    if objectives.violation <= 1e-9:
        lines.append("✓ No constraint violations -- this is a fully feasible plan.")
    else:
        lines.append(f"✗ Unresolved constraint violation (score {objectives.violation:.3f}) -- this was the least-infeasible plan found, not a feasible one.")

    feasible_entries = [e for e in archive.entries if e.objectives.violation <= 1e-9]
    if len(feasible_entries) > 1:
        cheaper = sum(1 for e in feasible_entries if e.objectives.cost_usd < selected.objectives.cost_usd - 1e-6)
        pricier = sum(1 for e in feasible_entries if e.objectives.cost_usd > selected.objectives.cost_usd + 1e-6)
        lines.append(f"Among {len(feasible_entries)} feasible plans found, this one is cheaper than {pricier} of them and more expensive than {cheaper} of them, based on your current priority weighting.")

    return lines


def compare_baseline_vs_optimized(problem: ProblemSpec, optimized_objectives: Objectives) -> dict:
    """Baseline = the greedy single-vessel-type-per-route heuristic
    (`benchmarking.baselines_optimization.run_greedy_heuristic`) -- already
    documented elsewhere in this platform as a stand-in for conventional
    route planning, so this reuses it rather than inventing a second
    baseline definition. Percentage improvement is only computed where
    mathematically valid (a nonzero baseline); otherwise None, never a
    fabricated number."""
    from quantumfleet.benchmarking.baselines_optimization import run_greedy_heuristic
    from quantumfleet.optimization.qea import evaluate

    baseline_plan = run_greedy_heuristic(problem)
    baseline_objectives = evaluate(baseline_plan, problem)

    def pct_improvement(baseline_value: float, optimized_value: float) -> float | None:
        if baseline_value == 0:
            return None
        return 100.0 * (baseline_value - optimized_value) / baseline_value

    return {
        "baseline_plan": baseline_plan,
        "baseline": baseline_objectives,
        "optimized": optimized_objectives,
        "fuel_improvement_pct": pct_improvement(baseline_objectives.fuel_tonnes, optimized_objectives.fuel_tonnes),
        "co2e_improvement_pct": pct_improvement(baseline_objectives.co2e_tonnes, optimized_objectives.co2e_tonnes),
        "cost_improvement_pct": pct_improvement(baseline_objectives.cost_usd, optimized_objectives.cost_usd),
    }


def pick_extreme_solutions(archive: ParetoArchive) -> dict[str, ArchiveEntry | None]:
    """The archive members that minimize each single objective, from the
    feasible subset when one exists. Used to let a user jump straight to
    "the greenest plan" or "the cheapest plan" without hand-tuning weights."""
    feasible = [e for e in archive.entries if e.objectives.violation <= 1e-9]
    pool = feasible if feasible else archive.entries
    if not pool:
        return {"lowest_fuel": None, "lowest_cost": None, "lowest_emissions": None}
    return {
        "lowest_fuel": min(pool, key=lambda e: e.objectives.fuel_tonnes),
        "lowest_cost": min(pool, key=lambda e: e.objectives.cost_usd),
        "lowest_emissions": min(pool, key=lambda e: e.objectives.co2e_tonnes),
    }


def explain_infeasibility(route_feasibilities: list[RouteFeasibility]) -> list[str]:
    """Plain-language reasons a plan failed to be fully feasible, derived
    directly from the same per-route feasibility breakdown shown in the
    Recommended Fleet Plan section -- never a separate, potentially
    inconsistent explanation. Suggestions name only the constraint actually
    measured as violated for that route, not a generic list."""
    reasons = []
    for r in route_feasibilities:
        if not r.has_any_assignment:
            reasons.append(f"**{r.route_id}**: no vessel was assigned to this route at all.")
            continue
        if r.cargo_fulfillment_pct < 99.9:
            reasons.append(
                f"**{r.route_id}**: cargo demand exceeds deliverable capacity -- "
                f"{r.cargo_capacity_tonnes:,.0f} t deliverable vs {r.cargo_demand_tonnes:,.0f} t required "
                f"({r.cargo_fulfillment_pct:.0f}% met). Consider a larger/faster compatible vessel, a longer "
                f"planning period, or lowering demand."
            )
        if not r.schedule_ok:
            reasons.append(
                f"**{r.route_id}**: schedule cannot be met -- the fastest assignment still takes "
                f"{r.worst_transit_days:.1f} days against a {r.max_transit_days:.1f}-day limit. Consider raising "
                f"the max transit time, or check whether a faster compatible vessel/fuel combination exists."
            )
        if not r.emission_cap_ok:
            over_pct = 100.0 * (r.route_co2e_tonnes - r.emission_cap_tonnes_co2e) / r.emission_cap_tonnes_co2e if r.emission_cap_tonnes_co2e > 0 else float("inf")
            reasons.append(
                f"**{r.route_id}**: emission cap exceeded -- {r.route_co2e_tonnes:,.0f} t CO2e against a cap of "
                f"{r.emission_cap_tonnes_co2e:,.0f} t (about {over_pct:.0f}% over). Consider raising the cap, "
                f"switching to a cleaner fuel or the green production pathway, or reducing speed."
            )
    if not reasons:
        reasons.append("No infeasibility found in the per-route breakdown for this plan.")
    return reasons
