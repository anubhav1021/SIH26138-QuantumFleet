"""Objective comparison, an external bounded Pareto archive with
crowding-distance trimming, and a Monte Carlo hypervolume estimator.

The archive is kept separate from the working QEA population
(`optimization.qea`): a population's own qubits can drift away from a good
solution it found earlier as the rotation-gate update keeps exploring, so
the archive is what actually accumulates the Pareto front across
generations. This is a deliberately simpler alternative to full NSGA-II
(no combined-population ranking/tournament needed every generation) that
still produces a genuine, checkable Pareto front.
"""

from dataclasses import dataclass, field

import numpy as np

from quantumfleet.optimization.encoding import FleetPlan


@dataclass(frozen=True)
class Objectives:
    fuel_tonnes: float
    co2e_tonnes: float
    cost_usd: float
    violation: float


def dominates(obj_a: Objectives, obj_b: Objectives, violation_tolerance: float = 1e-9) -> bool:
    """Constraint-dominance (Deb's rule): a feasible solution always beats an
    infeasible one; between two infeasible solutions, lower total violation
    wins; between two feasible solutions, standard Pareto dominance on
    (fuel, co2e, cost)."""
    a_feasible = obj_a.violation <= violation_tolerance
    b_feasible = obj_b.violation <= violation_tolerance

    if a_feasible != b_feasible:
        return a_feasible
    if not a_feasible:
        return obj_a.violation < obj_b.violation

    a_vals = (obj_a.fuel_tonnes, obj_a.co2e_tonnes, obj_a.cost_usd)
    b_vals = (obj_b.fuel_tonnes, obj_b.co2e_tonnes, obj_b.cost_usd)
    not_worse_anywhere = all(a <= b for a, b in zip(a_vals, b_vals))
    strictly_better_somewhere = any(a < b for a, b in zip(a_vals, b_vals))
    return not_worse_anywhere and strictly_better_somewhere


@dataclass
class ArchiveEntry:
    plan: FleetPlan
    objectives: Objectives


def crowding_distance(entries: list[ArchiveEntry]) -> np.ndarray:
    n = len(entries)
    if n == 0:
        return np.array([])
    if n <= 2:
        return np.full(n, np.inf)

    distances = np.zeros(n)
    for key in ("fuel_tonnes", "co2e_tonnes", "cost_usd"):
        values = np.array([getattr(e.objectives, key) for e in entries])
        order = np.argsort(values)
        distances[order[0]] = np.inf
        distances[order[-1]] = np.inf
        value_range = values[order[-1]] - values[order[0]]
        if value_range <= 0:
            continue
        for i in range(1, n - 1):
            distances[order[i]] += (values[order[i + 1]] - values[order[i - 1]]) / value_range
    return distances


@dataclass
class ParetoArchive:
    max_size: int = 60
    entries: list[ArchiveEntry] = field(default_factory=list)

    def try_add(self, plan: FleetPlan, objectives: Objectives) -> None:
        for e in self.entries:
            if dominates(e.objectives, objectives):
                return
        self.entries = [e for e in self.entries if not dominates(objectives, e.objectives)]
        self.entries.append(ArchiveEntry(plan=plan, objectives=objectives))
        if len(self.entries) > self.max_size:
            self.trim_by_crowding()

    def trim_by_crowding(self) -> None:
        while len(self.entries) > self.max_size:
            distances = crowding_distance(self.entries)
            self.entries.pop(int(np.argmin(distances)))

    def random_leader(self, rng: np.random.Generator) -> FleetPlan:
        idx = int(rng.integers(0, len(self.entries)))
        return self.entries[idx].plan

    def is_non_dominated_set(self) -> bool:
        """Test/debug helper: verifies no entry in the archive dominates another."""
        return all(
            not dominates(e1.objectives, e2.objectives)
            for i, e1 in enumerate(self.entries)
            for j, e2 in enumerate(self.entries)
            if i != j
        )

    def __len__(self) -> int:
        return len(self.entries)


def monte_carlo_hypervolume(
    archive: ParetoArchive,
    reference_point: tuple[float, float, float] | None = None,
    n_samples: int = 20_000,
    rng: np.random.Generator | None = None,
) -> float:
    """Monte Carlo estimate (n_samples draws, stated explicitly since this is
    an estimate, not an exact geometric computation) of the dominated
    hypervolume of the archive's feasible entries, against a fixed reference
    point (default: each objective's worst archive value x1.1)."""
    feasible = [e for e in archive.entries if e.objectives.violation <= 1e-9]
    if not feasible:
        return 0.0
    if rng is None:
        rng = np.random.default_rng(0)

    points = np.array([[e.objectives.fuel_tonnes, e.objectives.co2e_tonnes, e.objectives.cost_usd] for e in feasible])
    mins = points.min(axis=0)
    if reference_point is None:
        maxs = points.max(axis=0) * 1.1
        maxs = np.where(maxs > mins, maxs, mins + 1.0)
    else:
        maxs = np.array(reference_point)

    samples = rng.uniform(mins, maxs, size=(n_samples, 3))
    dominated = np.zeros(n_samples, dtype=bool)
    for p in points:
        dominated |= np.all(samples >= p, axis=1)

    volume_box = float(np.prod(maxs - mins))
    return float(dominated.mean()) * volume_box


def update_reference_point(reference_point: np.ndarray | None, archive: ParetoArchive, margin: float = 1.1) -> np.ndarray | None:
    """Tracks a reference point across generations that only ever grows
    (gets worse), never shrinks. Used to compute a comparable hypervolume
    series across a run for a convergence chart.

    `monte_carlo_hypervolume`'s default reference point is recomputed fresh
    from the CURRENT archive's own worst feasible value every call -- fine
    for a one-off hypervolume number, but wrong for tracking hypervolume
    generation-over-generation: as a genuinely improving, tightening Pareto
    front's worst point gets better, that shrinking reference point shrinks
    the measured box right along with it, so a real improvement can show up
    as a spuriously DECREASING hypervolume. Passing this running reference to
    `monte_carlo_hypervolume` instead keeps the box comparable across the
    whole run, so the chart reflects the front's own volume, not the
    reference chasing it."""
    feasible = [e for e in archive.entries if e.objectives.violation <= 1e-9]
    if not feasible:
        return reference_point
    current_worst = np.array(
        [
            max(e.objectives.fuel_tonnes for e in feasible),
            max(e.objectives.co2e_tonnes for e in feasible),
            max(e.objectives.cost_usd for e in feasible),
        ]
    ) * margin
    return current_worst if reference_point is None else np.maximum(reference_point, current_worst)
