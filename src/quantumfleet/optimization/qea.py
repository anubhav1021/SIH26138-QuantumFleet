"""Core quantum-inspired multi-objective evolutionary optimizer (QEA) for
green fleet deployment.

The fitness function calls the physics formula (`prediction.physics_model`)
directly, never the trained ML model: this keeps the inner loop fast and
deterministic (thousands of evaluations per run) and keeps the prediction
and optimization deliverables independently correct. The ML model is
instead used once, AFTER optimization, to cross-check the winning plan (see
`scripts/run_optimization.py`) -- "physics predicts X t, ML predicts Y t".
"""

from dataclasses import dataclass

import numpy as np

from quantumfleet.constants import DEFAULT_BERTH_TIME_DAYS
from quantumfleet.fuels import properties as fp
from quantumfleet.optimization import constraints as con
from quantumfleet.optimization.encoding import (
    Assignment,
    ChromosomeLayout,
    FleetPlan,
    QubitChromosome,
    assignment_to_gene_values,
    build_layout,
    decode,
)
from quantumfleet.optimization.pareto import Objectives, ParetoArchive, monte_carlo_hypervolume, update_reference_point
from quantumfleet.optimization.problem import ProblemSpec, Route
from quantumfleet.optimization.repair import repair_plan
from quantumfleet.prediction.physics_model import VoyageConditions, auxiliary_fuel_consumption_at_berth, predict_fuel_consumption_physics

DEFAULT_LOAD_FACTOR_ASSUMPTION = 0.85
THETA_MIN = 0.02  # radians; keeps cos^2(theta) within roughly [4e-4, 0.9996], avoiding premature convergence


def evaluate_single_assignment(a: Assignment, route: Route, problem: ProblemSpec) -> Objectives | None:
    """One assignment's own fuel/CO2e/cost contribution -- one round trip
    (two legs + one berth call), scaled by how many round trips fit in the
    planning period and by fleet count. Returns None if the assignment
    contributes nothing (zero count, or too slow to complete even one round
    trip this period). `violation` is always 0 here: constraint violation
    (cargo/schedule/emission) is only meaningful aggregated per route, not
    for one assignment in isolation -- see `evaluate_by_route`.

    Shared by `evaluate_by_route` (sums these per route) and
    `reporting.report_builder.per_assignment_breakdown` (reports them
    individually, since a route can hold several very differently-sized
    assignments -- see `optimization.encoding`'s multi-slot design)."""
    if a.count <= 0:
        return None
    trips = con.trips_per_period(route, a)
    if trips <= 0:
        return None

    scenario = problem.scenario
    vessel = problem.vessel_classes[a.vessel_class]
    berth_hours = DEFAULT_BERTH_TIME_DAYS * 24.0

    conditions = VoyageConditions(speed_knots=a.speed_knots, load_factor=DEFAULT_LOAD_FACTOR_ASSUMPTION, distance_nm=route.distance_nm)
    leg_pred = predict_fuel_consumption_physics(vessel, conditions, a.fuel_type)
    berth_pred = auxiliary_fuel_consumption_at_berth(vessel, berth_hours, a.fuel_type, a.use_shore_power)

    trip_fuel_tonnes = 2 * leg_pred.fuel_tonnes + berth_pred.fuel_tonnes
    trip_co2e_tonnes = 2 * leg_pred.co2e_tonnes + berth_pred.co2e_tonnes

    leg_fuel_cost = fp.fuel_mass_to_cost_usd(2 * leg_pred.fuel_tonnes * 1000.0, a.fuel_type, green=scenario.green_fuel_pathway)
    if a.use_shore_power:
        berth_cost = fp.shore_power_cost_usd(vessel.aux_load_kw * berth_hours)
    else:
        berth_cost = fp.fuel_mass_to_cost_usd(berth_pred.fuel_tonnes * 1000.0, a.fuel_type, green=scenario.green_fuel_pathway)
    charter_cost = vessel.day_rate_usd * route.period_days * a.count

    fleet_multiplier = trips * a.count
    return Objectives(
        fuel_tonnes=trip_fuel_tonnes * fleet_multiplier,
        co2e_tonnes=trip_co2e_tonnes * fleet_multiplier,
        cost_usd=(leg_fuel_cost + berth_cost) * fleet_multiplier + charter_cost,
        violation=0.0,
    )


def evaluate_by_route(plan: FleetPlan, problem: ProblemSpec) -> dict[str, Objectives]:
    """Per-route fuel/CO2e/cost/violation breakdown. Split out from
    `evaluate()` (which just sums this across routes) because
    `QuantumEvolutionaryOptimizer` also uses the per-route breakdown to track
    each route's best-known sub-solution independently and recombine them
    into a composite candidate -- see the class docstring for why."""
    by_route: dict[str, Objectives] = {}

    for route in problem.scenario.routes:
        assignments = [a for a in plan.assignments if a.route_id == route.route_id]
        route_fuel = route_co2e = route_cost = 0.0

        for a in assignments:
            result = evaluate_single_assignment(a, route, problem)
            if result is None:
                continue
            route_fuel += result.fuel_tonnes
            route_co2e += result.co2e_tonnes
            route_cost += result.cost_usd

        route_cost += problem.scenario.carbon_price_usd_per_tonne * route_co2e
        violation = (
            con.cargo_demand_violation(route, assignments, problem.vessel_classes)
            + con.schedule_violation(route, assignments)
            + con.emission_cap_violation(route, route_co2e)
        )
        by_route[route.route_id] = Objectives(fuel_tonnes=route_fuel, co2e_tonnes=route_co2e, cost_usd=route_cost, violation=violation)

    return by_route


def _aggregate_objectives(by_route: dict[str, Objectives]) -> Objectives:
    return Objectives(
        fuel_tonnes=sum(r.fuel_tonnes for r in by_route.values()),
        co2e_tonnes=sum(r.co2e_tonnes for r in by_route.values()),
        cost_usd=sum(r.cost_usd for r in by_route.values()),
        violation=sum(r.violation for r in by_route.values()),
    )


def evaluate(plan: FleetPlan, problem: ProblemSpec) -> Objectives:
    return _aggregate_objectives(evaluate_by_route(plan, problem))


def _rotate_qubit(alpha: float, beta: float, target_bit: int, delta_theta: float) -> tuple[float, float]:
    theta = float(np.arctan2(beta, alpha))
    theta += delta_theta if target_bit == 1 else -delta_theta
    theta = float(np.clip(theta, THETA_MIN, np.pi / 2 - THETA_MIN))
    return float(np.cos(theta)), float(np.sin(theta))


def rotate_towards(chromosome: QubitChromosome, leader_plan: FleetPlan, layout: ChromosomeLayout, delta_theta: float) -> None:
    """Rotates every qubit toward the corresponding choice in a concrete
    leader plan pulled from the Pareto archive (not a re-decoded/re-sampled
    version of the leader's own chromosome -- rotating toward an actual good
    solution is the standard Han-Kim QEA interpretation and avoids
    reintroducing sampling noise into "what the leader prefers").

    Relies on `leader_plan.assignments` being ordered exactly as `decode()`
    produces them (route_idx, then slot_idx) -- see test_encoding.py, which
    locks this invariant in."""
    for slot_position, leader_assignment in enumerate(leader_plan.assignments):
        route_idx, slot_idx = divmod(slot_position, layout.n_slots_per_route)
        base = layout.slot_offset(route_idx, slot_idx)
        gene_values = assignment_to_gene_values(leader_assignment, layout)

        for group in layout.gene_groups:
            start = base + group.offset
            if group.kind == "categorical":
                chosen = gene_values[group.name]
                for k in range(group.n_qubits):
                    idx = start + k
                    target_bit = 1 if k == chosen else 0
                    chromosome.alpha[idx], chromosome.beta[idx] = _rotate_qubit(chromosome.alpha[idx], chromosome.beta[idx], target_bit, delta_theta)
            elif group.kind == "binary":
                idx = start
                chromosome.alpha[idx], chromosome.beta[idx] = _rotate_qubit(chromosome.alpha[idx], chromosome.beta[idx], gene_values[group.name], delta_theta)
            elif group.kind == "count":
                value = gene_values[group.name]
                bits = [(value >> shift) & 1 for shift in reversed(range(group.n_qubits))]
                for m, bit in enumerate(bits):
                    idx = start + m
                    chromosome.alpha[idx], chromosome.beta[idx] = _rotate_qubit(chromosome.alpha[idx], chromosome.beta[idx], bit, delta_theta)


def quantum_mutate(chromosome: QubitChromosome, mutation_rate: float, rng: np.random.Generator) -> None:
    """Quantum NOT gate: swap (alpha, beta) for a randomly chosen subset of qubits."""
    mask = rng.random(len(chromosome.alpha)) < mutation_rate
    chromosome.alpha[mask], chromosome.beta[mask] = chromosome.beta[mask].copy(), chromosome.alpha[mask].copy()


@dataclass
class GenerationStats:
    generation: int
    archive_size: int
    hypervolume: float
    best_fuel_tonnes: float
    best_co2e_tonnes: float
    best_cost_usd: float


@dataclass
class QuantumEvolutionaryOptimizer:
    problem: ProblemSpec
    population_size: int = 40
    n_generations: int = 150
    delta_theta: float = 0.03 * np.pi
    mutation_rate: float = 0.02
    archive_max_size: int = 60
    catastrophe_fraction: float = 0.3
    stagnation_patience: int = 15
    seed: int = 42

    def run(self) -> tuple[ParetoArchive, list[GenerationStats]]:
        """Two mechanisms keep this from getting stuck, both addressing the
        same root cause: routes have independent qubits, but the archive's
        constraint-dominance is a SCALAR sum across routes.

        1. Composite-leader recombination. Before any feasible plan exists,
        constraint-dominance is a strict total order on violation, so the
        archive collapses to a single entry -- an individual needs EVERY
        route right simultaneously to improve on it, and that probability
        compounds badly as route count grows even though routes don't
        actually interact. Each generation, this tracks the best-known
        sub-solution independently PER ROUTE (possibly found by different
        individuals) and recombines them into one composite plan -- an
        implicit crossover across routes, the natural independent building
        blocks here -- which reliably reaches feasibility across many routes
        where waiting for one lucky individual would not.

        2. A "catastrophe" operator. Once the archive DOES hold a leader (or
        collapses to one), every individual rotates toward that same leader
        every generation, converging the whole population onto one point. This
        is the standard, well-documented QEA fix: when the best
        (min-violation, then max-hypervolume) score hasn't improved for
        `stagnation_patience` generations, reinitialize a fraction of the
        population back to fresh full superposition."""
        rng = np.random.default_rng(self.seed)
        layout = build_layout(self.problem)
        population = [QubitChromosome.initialize(layout) for _ in range(self.population_size)]
        archive = ParetoArchive(max_size=self.archive_max_size)
        history: list[GenerationStats] = []
        routes = self.problem.scenario.routes
        n_slots = layout.n_slots_per_route

        best_score_ever: tuple[float, float] | None = None
        stagnant_generations = 0
        best_by_route: dict[str, tuple[tuple[float, float], tuple, Objectives]] = {}
        reference_point: np.ndarray | None = None

        for gen in range(self.n_generations):
            for chromosome in population:
                plan = repair_plan(decode(chromosome, layout, rng), self.problem)
                by_route = evaluate_by_route(plan, self.problem)

                for route_idx, route in enumerate(routes):
                    route_obj = by_route[route.route_id]
                    score = (route_obj.violation, route_obj.fuel_tonnes)
                    if route.route_id not in best_by_route or score < best_by_route[route.route_id][0]:
                        route_assignments = plan.assignments[route_idx * n_slots : (route_idx + 1) * n_slots]
                        best_by_route[route.route_id] = (score, route_assignments, route_obj)

                archive.try_add(plan, _aggregate_objectives(by_route))

            if len(best_by_route) == len(routes):
                composite_assignments = tuple(a for route in routes for a in best_by_route[route.route_id][1])
                composite_by_route = {route.route_id: best_by_route[route.route_id][2] for route in routes}
                archive.try_add(FleetPlan(assignments=composite_assignments), _aggregate_objectives(composite_by_route))

            min_violation = min(e.objectives.violation for e in archive.entries)
            reference_point = update_reference_point(reference_point, archive)
            hv = monte_carlo_hypervolume(archive, reference_point=tuple(reference_point) if reference_point is not None else None, rng=rng)
            current_score = (min_violation, -hv)

            if best_score_ever is None or current_score < best_score_ever:
                best_score_ever = current_score
                stagnant_generations = 0
            else:
                stagnant_generations += 1

            if stagnant_generations >= self.stagnation_patience:
                n_reset = max(1, round(self.population_size * self.catastrophe_fraction))
                for idx in rng.choice(self.population_size, size=n_reset, replace=False):
                    population[idx] = QubitChromosome.initialize(layout)
                stagnant_generations = 0

            for chromosome in population:
                leader_plan = archive.random_leader(rng)
                rotate_towards(chromosome, leader_plan, layout, self.delta_theta)
                quantum_mutate(chromosome, self.mutation_rate, rng)

            history.append(
                GenerationStats(
                    generation=gen,
                    archive_size=len(archive),
                    hypervolume=hv,
                    best_fuel_tonnes=min(e.objectives.fuel_tonnes for e in archive.entries),
                    best_co2e_tonnes=min(e.objectives.co2e_tonnes for e in archive.entries),
                    best_cost_usd=min(e.objectives.cost_usd for e in archive.entries),
                )
            )

        return archive, history
