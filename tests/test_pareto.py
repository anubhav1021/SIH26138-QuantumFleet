import numpy as np

from quantumfleet.optimization.encoding import FleetPlan
from quantumfleet.optimization.pareto import ArchiveEntry, Objectives, ParetoArchive, crowding_distance, dominates, monte_carlo_hypervolume, update_reference_point

EMPTY_PLAN = FleetPlan(assignments=())


def obj(fuel, co2e, cost, violation=0.0):
    return Objectives(fuel_tonnes=fuel, co2e_tonnes=co2e, cost_usd=cost, violation=violation)


def test_feasible_always_beats_infeasible():
    feasible = obj(100, 100, 100, violation=0.0)
    infeasible = obj(1, 1, 1, violation=0.5)
    assert dominates(feasible, infeasible)
    assert not dominates(infeasible, feasible)


def test_lower_violation_wins_among_infeasible():
    less_bad = obj(10, 10, 10, violation=0.1)
    more_bad = obj(10, 10, 10, violation=0.5)
    assert dominates(less_bad, more_bad)
    assert not dominates(more_bad, less_bad)


def test_standard_pareto_dominance_among_feasible():
    better = obj(10, 10, 10)
    worse = obj(20, 20, 20)
    assert dominates(better, worse)
    assert not dominates(worse, better)


def test_non_dominated_points_do_not_dominate_each_other():
    a = obj(10, 20, 30)
    b = obj(20, 10, 30)
    assert not dominates(a, b)
    assert not dominates(b, a)


def test_archive_keeps_only_non_dominated_entries():
    archive = ParetoArchive(max_size=10)
    archive.try_add(EMPTY_PLAN, obj(20, 20, 20))
    archive.try_add(EMPTY_PLAN, obj(10, 10, 10))
    assert len(archive) == 1
    assert archive.entries[0].objectives.fuel_tonnes == 10


def test_archive_rejects_dominated_candidate():
    archive = ParetoArchive(max_size=10)
    archive.try_add(EMPTY_PLAN, obj(10, 10, 10))
    archive.try_add(EMPTY_PLAN, obj(20, 20, 20))
    assert len(archive) == 1


def test_archive_keeps_mutually_non_dominated_entries():
    archive = ParetoArchive(max_size=10)
    archive.try_add(EMPTY_PLAN, obj(10, 20, 30))
    archive.try_add(EMPTY_PLAN, obj(20, 10, 30))
    assert len(archive) == 2
    assert archive.is_non_dominated_set()


def test_crowding_distance_hand_crafted():
    entries = [
        ArchiveEntry(EMPTY_PLAN, obj(0, 10, 0)),
        ArchiveEntry(EMPTY_PLAN, obj(5, 5, 5)),
        ArchiveEntry(EMPTY_PLAN, obj(10, 0, 10)),
    ]
    distances = crowding_distance(entries)
    assert distances[0] == np.inf
    assert distances[2] == np.inf
    assert np.isfinite(distances[1])


def test_trim_by_crowding_removes_least_crowded_point():
    archive = ParetoArchive(max_size=4)
    for fuel, co2e, cost in [(0.0, 30.0, 5.0), (1.0, 12.0, 5.0), (2.0, 6.0, 5.0), (10.0, 0.0, 5.0)]:
        archive.try_add(EMPTY_PLAN, obj(fuel, co2e, cost))
    assert len(archive) == 4

    archive.max_size = 3
    archive.trim_by_crowding()

    remaining_fuels = sorted(e.objectives.fuel_tonnes for e in archive.entries)
    assert remaining_fuels == [0.0, 2.0, 10.0]


def test_hypervolume_prefers_better_front():
    small_win = ParetoArchive(max_size=10)
    small_win.try_add(EMPTY_PLAN, obj(10, 10, 10))

    big_win = ParetoArchive(max_size=10)
    big_win.try_add(EMPTY_PLAN, obj(50, 50, 50))

    hv_small = monte_carlo_hypervolume(small_win, reference_point=(100, 100, 100), rng=np.random.default_rng(0))
    hv_big = monte_carlo_hypervolume(big_win, reference_point=(100, 100, 100), rng=np.random.default_rng(0))
    assert hv_small > hv_big


def test_hypervolume_zero_with_no_feasible_entries():
    archive = ParetoArchive(max_size=10)
    archive.try_add(EMPTY_PLAN, obj(10, 10, 10, violation=1.0))
    assert monte_carlo_hypervolume(archive) == 0.0


def test_reference_point_never_shrinks_as_front_improves():
    """A tightening, genuinely improving Pareto front must not shrink the
    tracked reference point -- otherwise hypervolume computed against it can
    spuriously decrease even though the front got better (see
    pareto.update_reference_point's docstring)."""
    archive = ParetoArchive(max_size=10)
    archive.try_add(EMPTY_PLAN, obj(100, 100, 100))
    ref1 = update_reference_point(None, archive)

    # replace with a strictly better (lower) point: front improved, reference must not shrink
    archive = ParetoArchive(max_size=10)
    archive.try_add(EMPTY_PLAN, obj(10, 10, 10))
    ref2 = update_reference_point(ref1, archive)

    assert np.all(ref2 >= ref1)


def test_reference_point_grows_to_cover_a_worse_point():
    archive = ParetoArchive(max_size=10)
    archive.try_add(EMPTY_PLAN, obj(10, 10, 10))
    ref1 = update_reference_point(None, archive)

    archive.try_add(EMPTY_PLAN, obj(5, 5, 200))  # non-dominated: worse cost, better fuel/co2e
    ref2 = update_reference_point(ref1, archive)

    assert ref2[2] > ref1[2]  # cost component grew to cover the new worse value


def test_reference_point_ignores_infeasible_entries():
    archive = ParetoArchive(max_size=10)
    archive.try_add(EMPTY_PLAN, obj(10, 10, 10, violation=5.0))  # only entry is infeasible: no feasible baseline to reference yet
    assert update_reference_point(None, archive) is None
