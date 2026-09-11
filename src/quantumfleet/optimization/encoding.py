"""Qubit encoding/decoding for the quantum-inspired fleet optimizer.

Decision unit = an "assignment slot": each route can host up to
`problem.n_slots_per_route` independent vessel/fuel combinations (a mixed
fleet on one route), giving the optimizer room to split a route's cargo
across, e.g., a mostly-HFO fleet plus a smaller green-methanol vessel.

Each slot has five gene groups. Three are categorical (vessel_class,
fuel_type, speed_bin) and share ONE decode mechanism (roulette-wheel
observation) -- the trick that makes a mixed categorical/integer decision
space tractable without inventing a different mechanism per group. `count`
uses the classic Han-Kim binary-positional QEA encoding. `use_shore_power` is
a single Bernoulli-observed qubit, kept separate from fuel_type because
shore power displaces at-berth *auxiliary* load with grid electricity -- it
says nothing about propulsion-fuel choice during transit.

`speed_bin`'s ordinality is deliberately NOT encoded in the qubits (no Gray
coding): the rotation-gate update and mutation don't need adjacency
structure to work, and ordinality is preserved exactly where it matters --
in decode (bin index -> knot value) and evaluation -- so nothing about the
search is broken by treating it as an unordered category.

This module only decodes; it does not repair/validate the result (see
`optimization.repair.repair_plan`), keeping decode a pure function of the
chromosome and the caller free to compose decode + repair explicitly.
"""

from dataclasses import dataclass, field

import numpy as np

from quantumfleet.fuels.properties import PROPULSION_FUELS
from quantumfleet.optimization.problem import ProblemSpec

COUNT_BITS = 3
COUNT_MAX = 2**COUNT_BITS - 1  # 7


@dataclass(frozen=True)
class Assignment:
    route_id: str
    vessel_class: str
    fuel_type: str
    speed_knots: float
    use_shore_power: bool
    count: int


@dataclass(frozen=True)
class FleetPlan:
    assignments: tuple[Assignment, ...]


@dataclass(frozen=True)
class GeneGroup:
    name: str
    kind: str  # "categorical" | "binary" | "count"
    n_qubits: int
    offset: int  # start index within ONE slot's qubit block


@dataclass(frozen=True)
class ChromosomeLayout:
    problem: ProblemSpec
    vessel_class_ids: tuple[str, ...]
    fuel_type_ids: tuple[str, ...]
    gene_groups: tuple[GeneGroup, ...]
    qubits_per_slot: int
    n_slots_per_route: int
    n_routes: int

    @property
    def total_qubits(self) -> int:
        return self.qubits_per_slot * self.n_slots_per_route * self.n_routes

    def slot_offset(self, route_idx: int, slot_idx: int) -> int:
        return (route_idx * self.n_slots_per_route + slot_idx) * self.qubits_per_slot


def build_layout(problem: ProblemSpec) -> ChromosomeLayout:
    vessel_class_ids = tuple(problem.vessel_classes.keys())
    fuel_type_ids = tuple(PROPULSION_FUELS)

    groups = []
    offset = 0
    for name, kind, n in [
        ("vessel_class", "categorical", len(vessel_class_ids)),
        ("fuel_type", "categorical", len(fuel_type_ids)),
        ("speed_bin", "categorical", len(problem.speed_bins_knots)),
        ("use_shore_power", "binary", 1),
        ("count", "count", COUNT_BITS),
    ]:
        groups.append(GeneGroup(name=name, kind=kind, n_qubits=n, offset=offset))
        offset += n

    return ChromosomeLayout(
        problem=problem,
        vessel_class_ids=vessel_class_ids,
        fuel_type_ids=fuel_type_ids,
        gene_groups=tuple(groups),
        qubits_per_slot=offset,
        n_slots_per_route=problem.n_slots_per_route,
        n_routes=len(problem.scenario.routes),
    )


@dataclass
class QubitChromosome:
    alpha: np.ndarray
    beta: np.ndarray

    @staticmethod
    def initialize(layout: ChromosomeLayout) -> "QubitChromosome":
        """Equal superposition (alpha=beta=1/sqrt(2)) for every qubit -- the
        standard QEA starting point, giving every option equal initial
        probability and maximal room to explore."""
        n = layout.total_qubits
        inv_sqrt2 = 1.0 / np.sqrt(2.0)
        return QubitChromosome(alpha=np.full(n, inv_sqrt2), beta=np.full(n, inv_sqrt2))


def observe_categorical(alpha: np.ndarray, beta: np.ndarray, rng: np.random.Generator) -> int:
    """Roulette-wheel observation over len(alpha) mutually-exclusive options:
    p_j = alpha_j^2 / sum(alpha_i^2)."""
    probs = alpha**2
    total = probs.sum()
    probs = probs / total if total > 0 else np.full(len(alpha), 1.0 / len(alpha))
    return int(rng.choice(len(alpha), p=probs))


def observe_binary(alpha: float, beta: float, rng: np.random.Generator) -> int:
    return int(rng.random() < beta**2)


def observe_count(alphas: np.ndarray, betas: np.ndarray, rng: np.random.Generator) -> int:
    value = 0
    for a, b in zip(alphas, betas):
        value = (value << 1) | observe_binary(a, b, rng)
    return value


def decode(chromosome: QubitChromosome, layout: ChromosomeLayout, rng: np.random.Generator) -> FleetPlan:
    """Pure decode: does not repair/validate feasibility (see
    `optimization.repair.repair_plan`, meant to be composed by the caller).
    Produces assignments in (route_idx, then slot_idx) order -- this order is
    relied on by `optimization.qea.rotate_towards` to pair a leader plan's
    assignments back to their qubit positions; see test_encoding.py."""
    problem = layout.problem
    assignments = []
    for route_idx, route in enumerate(problem.scenario.routes):
        for slot_idx in range(layout.n_slots_per_route):
            base = layout.slot_offset(route_idx, slot_idx)
            values: dict[str, int] = {}
            for group in layout.gene_groups:
                start = base + group.offset
                end = start + group.n_qubits
                a = chromosome.alpha[start:end]
                b = chromosome.beta[start:end]
                if group.kind == "categorical":
                    values[group.name] = observe_categorical(a, b, rng)
                elif group.kind == "binary":
                    values[group.name] = observe_binary(a[0], b[0], rng)
                elif group.kind == "count":
                    values[group.name] = observe_count(a, b, rng)

            assignments.append(
                Assignment(
                    route_id=route.route_id,
                    vessel_class=layout.vessel_class_ids[values["vessel_class"]],
                    fuel_type=layout.fuel_type_ids[values["fuel_type"]],
                    speed_knots=problem.speed_bins_knots[values["speed_bin"]],
                    use_shore_power=bool(values["use_shore_power"]),
                    count=values["count"],
                )
            )
    return FleetPlan(assignments=tuple(assignments))


def assignment_to_gene_values(assignment: Assignment, layout: ChromosomeLayout) -> dict[str, int]:
    """Inverse of one slot's decode: recovers the gene-group index values
    (which option was chosen) for an already-decoded Assignment. Used by
    `optimization.qea.rotate_towards` to rotate a population's qubits toward
    a concrete leader plan pulled from the Pareto archive."""
    return {
        "vessel_class": layout.vessel_class_ids.index(assignment.vessel_class),
        "fuel_type": layout.fuel_type_ids.index(assignment.fuel_type),
        "speed_bin": layout.problem.speed_bins_knots.index(assignment.speed_knots),
        "use_shore_power": int(assignment.use_shore_power),
        "count": assignment.count,
    }
