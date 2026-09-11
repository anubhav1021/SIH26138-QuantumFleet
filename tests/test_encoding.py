import numpy as np

from quantumfleet.optimization.encoding import (
    COUNT_MAX,
    QubitChromosome,
    assignment_to_gene_values,
    build_layout,
    decode,
    observe_categorical,
)
from quantumfleet.optimization.problem import load_problem


def _problem(vessel_catalog_path, default_scenario_path):
    return load_problem(default_scenario_path, vessel_catalog_path)


def test_forced_collapse_decodes_deterministically(vessel_catalog_path, default_scenario_path):
    problem = _problem(vessel_catalog_path, default_scenario_path)
    layout = build_layout(problem)
    rng = np.random.default_rng(0)

    chromosome = QubitChromosome.initialize(layout)
    base = layout.slot_offset(0, 0)
    for group in layout.gene_groups:
        start = base + group.offset
        end = start + group.n_qubits
        chromosome.alpha[start:end] = 0.0
        chromosome.beta[start:end] = 1.0
        if group.kind == "categorical":
            chromosome.alpha[start] = 1.0
            chromosome.beta[start] = 0.0

    for _ in range(20):
        plan = decode(chromosome, layout, rng)
        first = plan.assignments[0]
        assert first.vessel_class == layout.vessel_class_ids[0]
        assert first.fuel_type == layout.fuel_type_ids[0]
        assert first.speed_knots == problem.speed_bins_knots[0]
        assert first.use_shore_power is True
        assert first.count == COUNT_MAX


def test_full_superposition_decodes_approximately_uniformly(vessel_catalog_path, default_scenario_path):
    problem = _problem(vessel_catalog_path, default_scenario_path)
    layout = build_layout(problem)
    rng = np.random.default_rng(1)
    chromosome = QubitChromosome.initialize(layout)

    vessel_group = layout.gene_groups[0]
    base = layout.slot_offset(0, 0)
    start = base + vessel_group.offset
    end = start + vessel_group.n_qubits

    n_trials = 3000
    counts = np.zeros(vessel_group.n_qubits)
    for _ in range(n_trials):
        idx = observe_categorical(chromosome.alpha[start:end], chromosome.beta[start:end], rng)
        counts[idx] += 1

    expected = n_trials / vessel_group.n_qubits
    assert np.all(counts > expected * 0.5)
    assert np.all(counts < expected * 1.5)


def test_decoded_assignments_ordered_by_route_then_slot(vessel_catalog_path, default_scenario_path):
    problem = _problem(vessel_catalog_path, default_scenario_path)
    layout = build_layout(problem)
    rng = np.random.default_rng(2)
    chromosome = QubitChromosome.initialize(layout)
    plan = decode(chromosome, layout, rng)

    expected = [route.route_id for route in problem.scenario.routes for _ in range(layout.n_slots_per_route)]
    actual = [a.route_id for a in plan.assignments]
    assert actual == expected


def test_assignment_to_gene_values_round_trip(vessel_catalog_path, default_scenario_path):
    problem = _problem(vessel_catalog_path, default_scenario_path)
    layout = build_layout(problem)
    rng = np.random.default_rng(3)
    chromosome = QubitChromosome.initialize(layout)
    plan = decode(chromosome, layout, rng)

    for a in plan.assignments:
        values = assignment_to_gene_values(a, layout)
        assert layout.vessel_class_ids[values["vessel_class"]] == a.vessel_class
        assert layout.fuel_type_ids[values["fuel_type"]] == a.fuel_type
        assert problem.speed_bins_knots[values["speed_bin"]] == a.speed_knots
        assert bool(values["use_shore_power"]) == a.use_shore_power
        assert values["count"] == a.count


def test_total_qubits_matches_layout_arithmetic(vessel_catalog_path, default_scenario_path):
    problem = _problem(vessel_catalog_path, default_scenario_path)
    layout = build_layout(problem)
    n_vessel_classes = len(problem.vessel_classes)
    n_fuels = 6
    n_speed_bins = len(problem.speed_bins_knots)
    qubits_per_slot = n_vessel_classes + n_fuels + n_speed_bins + 1 + 3
    expected_total = qubits_per_slot * layout.n_slots_per_route * len(problem.scenario.routes)
    assert layout.total_qubits == expected_total
    chromosome = QubitChromosome.initialize(layout)
    assert len(chromosome.alpha) == expected_total
