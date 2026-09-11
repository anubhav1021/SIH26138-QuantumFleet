from dataclasses import replace
from pathlib import Path

from quantumfleet.optimization.problem import load_problem
from quantumfleet.optimization.qea import QuantumEvolutionaryOptimizer
from quantumfleet.reporting.report_builder import build_report


def test_build_report_produces_expected_files(tmp_path, vessel_catalog_path, default_scenario_path):
    problem = load_problem(default_scenario_path, vessel_catalog_path)
    small = replace(problem, scenario=replace(problem.scenario, routes=problem.scenario.routes[:2]))
    archive, history = QuantumEvolutionaryOptimizer(problem=small, population_size=15, n_generations=15, seed=1).run()

    report_path = build_report(small, archive, history, str(tmp_path))

    assert Path(report_path).exists()
    assert (tmp_path / "pareto_front.png").exists()
    assert (tmp_path / "convergence.png").exists()
    content = Path(report_path).read_text(encoding="utf-8")
    assert "Fleet allocation" in content
