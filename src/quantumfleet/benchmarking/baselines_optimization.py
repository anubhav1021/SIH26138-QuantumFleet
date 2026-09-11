"""Classical baselines for fleet-optimization benchmarking: random search, a
greedy single-vessel-type heuristic, and a real-coded genetic algorithm --
all operating on the SAME decoded Assignment representation and reusing
`optimization.repair` / `optimization.constraints` / `optimization.pareto`
unchanged. This isolates exactly what the quantum-inspired rotation-gate
mechanism contributes versus a conventional evolutionary search using the
same fitness, constraints, and archive.
"""

import math
from dataclasses import dataclass, replace

import numpy as np

from quantumfleet.fuels.properties import PROPULSION_FUELS
from quantumfleet.optimization import constraints as con
from quantumfleet.optimization.encoding import Assignment, FleetPlan
from quantumfleet.optimization.pareto import Objectives, ParetoArchive, dominates, monte_carlo_hypervolume, update_reference_point
from quantumfleet.optimization.problem import ProblemSpec
from quantumfleet.optimization.qea import GenerationStats, evaluate
from quantumfleet.optimization.repair import repair_plan


def random_assignment(route_id: str, problem: ProblemSpec, rng: np.random.Generator) -> Assignment:
    vessel_class = str(rng.choice(list(problem.vessel_classes.keys())))
    fuel_type = str(rng.choice(PROPULSION_FUELS))
    speed_knots = float(rng.choice(problem.speed_bins_knots))
    use_shore_power = bool(rng.random() < 0.5)
    count = int(rng.integers(0, 8))
    return Assignment(route_id=route_id, vessel_class=vessel_class, fuel_type=fuel_type, speed_knots=speed_knots, use_shore_power=use_shore_power, count=count)


def random_plan(problem: ProblemSpec, rng: np.random.Generator) -> FleetPlan:
    assignments = [
        random_assignment(route.route_id, problem, rng)
        for route in problem.scenario.routes
        for _ in range(problem.n_slots_per_route)
    ]
    return FleetPlan(assignments=tuple(assignments))


def _record_stats(
    generation: int, archive: ParetoArchive, reference_point: np.ndarray | None, rng: np.random.Generator
) -> tuple[GenerationStats, np.ndarray | None]:
    """Returns (stats, updated reference_point). See
    `pareto.update_reference_point` for why hypervolume is tracked against a
    running (only ever growing) reference rather than one recomputed fresh
    from the current archive every generation."""
    reference_point = update_reference_point(reference_point, archive)
    hv = monte_carlo_hypervolume(archive, reference_point=tuple(reference_point) if reference_point is not None else None, rng=rng)
    stats = GenerationStats(
        generation=generation,
        archive_size=len(archive),
        hypervolume=hv,
        best_fuel_tonnes=min(e.objectives.fuel_tonnes for e in archive.entries),
        best_co2e_tonnes=min(e.objectives.co2e_tonnes for e in archive.entries),
        best_cost_usd=min(e.objectives.cost_usd for e in archive.entries),
    )
    return stats, reference_point


def run_random_search(problem: ProblemSpec, population_size: int, n_generations: int, seed: int = 42) -> tuple[ParetoArchive, list[GenerationStats]]:
    """Same evaluation budget (population_size x n_generations plans) as the
    QEA/GA, spent on pure random sampling -- the floor any real search method
    should beat."""
    rng = np.random.default_rng(seed)
    archive = ParetoArchive(max_size=60)
    history = []
    reference_point: np.ndarray | None = None
    for gen in range(n_generations):
        for _ in range(population_size):
            plan = repair_plan(random_plan(problem, rng), problem)
            archive.try_add(plan, evaluate(plan, problem))
        stats, reference_point = _record_stats(gen, archive, reference_point, rng)
        history.append(stats)
    return archive, history


def run_greedy_heuristic(problem: ProblemSpec) -> FleetPlan:
    """For each route independently: try every (vessel_class, fuel_type,
    speed, shore_power) combination compatible with that route, size the
    count to just cover cargo demand, and keep the cheapest option among
    feasible candidates (least-violating if none are feasible). One vessel
    type per route, no mixed-fleet slots, no cross-route search -- a
    reasonable stand-in for conventional single-vessel-type route planning,
    deterministic and requiring no metaheuristic machinery at all."""
    assignments = []
    for route in problem.scenario.routes:
        single_route_problem = replace(problem, scenario=replace(problem.scenario, routes=(route,)))
        best_assignment = None
        best_score = None

        for vessel in problem.vessel_classes.values():
            allowed_fuels = [f for f in route.allowed_fuel_types if f in vessel.compatible_fuels]
            for fuel_type in allowed_fuels:
                for speed in problem.speed_bins_knots:
                    if not (vessel.min_speed_knots <= speed <= vessel.max_speed_knots):
                        continue
                    for shore_power in ([True, False] if route.shore_power_available else [False]):
                        probe = Assignment(route_id=route.route_id, vessel_class=vessel.id, fuel_type=fuel_type, speed_knots=speed, use_shore_power=shore_power, count=1)
                        trips = con.trips_per_period(route, probe)
                        if trips <= 0:
                            continue
                        capacity_per_count = vessel.dwt_tonnes * trips
                        count = min(7, max(1, math.ceil(route.cargo_demand_tonnes / capacity_per_count)))
                        candidate = replace(probe, count=count)

                        obj = evaluate(FleetPlan(assignments=(candidate,)), single_route_problem)
                        score = (obj.violation, obj.cost_usd)
                        if best_score is None or score < best_score:
                            best_score, best_assignment = score, candidate

        if best_assignment is not None:
            assignments.append(best_assignment)

    return FleetPlan(assignments=tuple(assignments))


def _tournament_select(evaluated: list[tuple[FleetPlan, Objectives]], k: int, rng: np.random.Generator) -> FleetPlan:
    idx = rng.integers(0, len(evaluated), size=k)
    best_plan, best_obj = evaluated[idx[0]]
    for i in idx[1:]:
        plan, obj = evaluated[i]
        if dominates(obj, best_obj):
            best_plan, best_obj = plan, obj
    return best_plan


def _uniform_crossover(parent1: FleetPlan, parent2: FleetPlan, rng: np.random.Generator) -> FleetPlan:
    assignments = [a1 if rng.random() < 0.5 else a2 for a1, a2 in zip(parent1.assignments, parent2.assignments)]
    return FleetPlan(assignments=tuple(assignments))


def _random_reset_mutate(plan: FleetPlan, problem: ProblemSpec, mutation_rate: float, rng: np.random.Generator) -> FleetPlan:
    assignments = [
        random_assignment(a.route_id, problem, rng) if rng.random() < mutation_rate else a
        for a in plan.assignments
    ]
    return FleetPlan(assignments=tuple(assignments))


@dataclass
class ClassicalGeneticAlgorithm:
    """Real-coded GA on the same decoded Assignment representation as the
    QEA: tournament selection, per-slot uniform crossover (an explicit,
    classical analogue of the QEA's composite-leader recombination -- both
    recombine independent per-route building blocks, just via different
    mechanisms), and random-reset mutation."""

    problem: ProblemSpec
    population_size: int = 40
    n_generations: int = 150
    crossover_rate: float = 0.9
    mutation_rate: float = 0.05
    tournament_size: int = 3
    archive_max_size: int = 60
    seed: int = 42

    def run(self) -> tuple[ParetoArchive, list[GenerationStats]]:
        rng = np.random.default_rng(self.seed)
        population = [repair_plan(random_plan(self.problem, rng), self.problem) for _ in range(self.population_size)]
        archive = ParetoArchive(max_size=self.archive_max_size)
        history = []
        reference_point: np.ndarray | None = None

        for gen in range(self.n_generations):
            evaluated = [(plan, evaluate(plan, self.problem)) for plan in population]
            for plan, obj in evaluated:
                archive.try_add(plan, obj)

            new_population = []
            while len(new_population) < self.population_size:
                parent1 = _tournament_select(evaluated, self.tournament_size, rng)
                parent2 = _tournament_select(evaluated, self.tournament_size, rng)
                child = _uniform_crossover(parent1, parent2, rng) if rng.random() < self.crossover_rate else parent1
                child = _random_reset_mutate(child, self.problem, self.mutation_rate, rng)
                new_population.append(repair_plan(child, self.problem))
            population = new_population

            stats, reference_point = _record_stats(gen, archive, reference_point, rng)
            history.append(stats)

        return archive, history
