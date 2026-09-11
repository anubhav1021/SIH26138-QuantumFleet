"""Assembles a Markdown report from a completed optimization run -- the
single source of truth for report layout, called by both
`scripts/run_case_study.py` and the dashboard's "export report" button.
"""

from pathlib import Path

import pandas as pd

from quantumfleet.optimization.encoding import FleetPlan
from quantumfleet.optimization.pareto import ParetoArchive
from quantumfleet.optimization.problem import ProblemSpec
from quantumfleet.optimization.qea import GenerationStats, evaluate_single_assignment
from quantumfleet.reporting.charts import save_static_convergence, save_static_pareto_front


def plan_to_dataframe(plan: FleetPlan) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "route_id": a.route_id,
                "vessel_class": a.vessel_class,
                "fuel_type": a.fuel_type,
                "speed_knots": a.speed_knots,
                "shore_power": a.use_shore_power,
                "count": a.count,
            }
            for a in plan.assignments
            if a.count > 0
        ]
    )


def per_assignment_breakdown(plan: FleetPlan, problem: ProblemSpec) -> pd.DataFrame:
    """Per-assignment fuel/CO2e/cost, not aggregated per route like
    `qea.evaluate_by_route` -- needed for reporting/visualization where each
    row's actual share matters, since a route can hold several very
    differently-sized assignments (see `optimization.encoding`'s multi-slot
    design). An emissions-by-fuel-type chart built from the configuration
    table alone (route/vessel/fuel/count, no computed values) would have to
    guess at each row's share; this gives the real number."""
    routes_by_id = {r.route_id: r for r in problem.scenario.routes}
    rows = []
    for a in plan.assignments:
        if a.count <= 0:
            continue
        result = evaluate_single_assignment(a, routes_by_id[a.route_id], problem)
        if result is None:
            continue
        rows.append(
            {
                "route_id": a.route_id,
                "vessel_class": a.vessel_class,
                "fuel_type": a.fuel_type,
                "speed_knots": a.speed_knots,
                "shore_power": a.use_shore_power,
                "count": a.count,
                "fuel_tonnes": result.fuel_tonnes,
                "co2e_tonnes": result.co2e_tonnes,
                "cost_usd": result.cost_usd,
            }
        )
    return pd.DataFrame(rows)


def _markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_(no active assignments)_"
    header = "| " + " | ".join(df.columns) + " |"
    separator = "| " + " | ".join("---" for _ in df.columns) + " |"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([header, separator, *rows])


def build_report(
    problem: ProblemSpec,
    archive: ParetoArchive,
    history: list[GenerationStats],
    output_dir: str,
    title: str = "Green Fleet Optimization Report",
    filename: str = "report.md",
    image_prefix: str = "",
) -> str:
    """Writes chart images and a Markdown report into `output_dir`; returns
    the report's own path. `image_prefix` lets multiple reports share one
    output_dir without their images colliding (e.g. "case_study_")."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pareto_image = f"{image_prefix}pareto_front.png"
    convergence_image = f"{image_prefix}convergence.png"
    save_static_pareto_front(archive, str(out / pareto_image))
    save_static_convergence({"Quantum-Inspired (QEA)": history}, str(out / convergence_image))

    feasible = [e for e in archive.entries if e.objectives.violation <= 1e-9]
    best = min(feasible, key=lambda e: e.objectives.cost_usd) if feasible else min(archive.entries, key=lambda e: e.objectives.violation)
    recommendation_heading = "## Recommended plan (lowest cost among feasible solutions)" if feasible else "## Best plan found (lowest constraint violation -- no fully feasible plan found)"

    lines = [
        f"# {title}",
        "",
        f"Scenario: **{problem.scenario.name}** -- {len(problem.scenario.routes)} routes, "
        f"carbon price ${problem.scenario.carbon_price_usd_per_tonne:.0f}/t CO2e, "
        f"{'green' if problem.scenario.green_fuel_pathway else 'grey'} alt-fuel pathway.",
        "",
        "## Optimization summary",
        "",
        f"- Pareto archive: **{len(archive)}** non-dominated plans, **{len(feasible)}** fully feasible.",
        f"- Final hypervolume: **{history[-1].hypervolume:,.1f}**" if history else "",
        "",
        f"![Pareto front]({pareto_image})",
        "",
        f"![Convergence]({convergence_image})",
        "",
        recommendation_heading,
        "",
        f"- Fuel: **{best.objectives.fuel_tonnes:,.1f} t**",
        f"- Lifecycle CO2e: **{best.objectives.co2e_tonnes:,.1f} t**",
        f"- Total cost: **${best.objectives.cost_usd:,.0f}**",
        f"- Constraint violation: **{best.objectives.violation:.4f}**",
        "",
        "### Fleet allocation",
        "",
        _markdown_table(per_assignment_breakdown(best.plan, problem).round(1)),
        "",
    ]

    report_path = out / filename
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return str(report_path)
