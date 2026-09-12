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
from quantumfleet.reporting.explain import compare_baseline_vs_optimized, explain_infeasibility, explain_why_plan_chosen, plan_summary_metrics


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
    summary = plan_summary_metrics(best.plan, best.objectives, problem)

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
        f"- Total cost: **${best.objectives.cost_usd:,.0f}** (of which carbon cost: ${summary['carbon_cost_usd']:,.0f})",
        f"- Cargo fulfillment: **{summary['cargo_fulfillment_pct']:.0f}%** average across routes",
        f"- Schedule met on **{summary['schedule_compliant_routes']}/{summary['n_routes']}** routes; emission cap respected on **{summary['emission_compliant_routes']}/{summary['n_routes']}** routes",
        f"- Vessels deployed: **{summary['vessels_used']}**; fuel mix: {', '.join(f'{fuel} x{count}' for fuel, count in summary['fuel_mix'].items()) or '(none)'}",
        f"- Constraint violation: **{best.objectives.violation:.4f}**",
        "",
        "### Fleet allocation",
        "",
        _markdown_table(per_assignment_breakdown(best.plan, problem).round(1)),
        "",
        "### Why this plan",
        "",
    ]
    lines += [f"- {line}" for line in explain_why_plan_chosen(best.objectives, summary["route_feasibilities"], archive, best)]

    if best.objectives.violation > 1e-9:
        lines += ["", "### Why it's infeasible", ""]
        lines += [f"- {reason}" for reason in explain_infeasibility(summary["route_feasibilities"])]

    baseline = compare_baseline_vs_optimized(problem, best.objectives)
    lines += ["", "### Baseline vs. optimized", "", "Baseline = a greedy, single-vessel-type-per-route heuristic (conventional route planning, no metaheuristic search).", ""]
    baseline_rows = []
    for label, key, pct_key in [("Fuel (t)", "fuel_tonnes", "fuel_improvement_pct"), ("CO2e (t)", "co2e_tonnes", "co2e_improvement_pct"), ("Cost (USD)", "cost_usd", "cost_improvement_pct")]:
        baseline_v, optimized_v, pct = getattr(baseline["baseline"], key), getattr(baseline["optimized"], key), baseline[pct_key]
        baseline_rows.append({"Metric": label, "Baseline (greedy)": round(baseline_v, 1), "Quantum-inspired": round(optimized_v, 1), "% improvement": f"{pct:+.1f}%" if pct is not None else "N/A"})
    lines.append(_markdown_table(pd.DataFrame(baseline_rows)))

    lines += [
        "",
        "### Important assumptions",
        "",
        "- Fuel properties (LHV, density, well-to-wake CO2e, indicative prices) are representative values assembled "
        "from standard maritime-engineering and IMO/EU GHG literature orders of magnitude -- internally consistent, "
        "but not citation-pinned. See `src/quantumfleet/fuels/properties.py`.",
        "- The physics model (Admiralty-coefficient method) is a long-standing empirical naval-architecture "
        "approximation, not a first-principles hydrodynamic computation.",
        "- No real voyage dataset was available for this problem statement; the fuel-prediction models are trained "
        "on a physics-informed synthetic generator (see `docs/implementation_guide.md`).",
        "",
    ]

    report_path = out / filename
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return str(report_path)
