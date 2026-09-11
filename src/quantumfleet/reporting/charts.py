"""Chart builders: matplotlib figures for static report export, plotly
figures for the interactive Streamlit dashboard. Both read directly from a
`ParetoArchive` / `list[GenerationStats]` / benchmark results dict so the
dashboard and the case-study report always show the same data shapes.
"""

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from quantumfleet.optimization.pareto import ParetoArchive
from quantumfleet.optimization.qea import GenerationStats

matplotlib.use("Agg")


def archive_to_dataframe(archive: ParetoArchive) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "fuel_tonnes": e.objectives.fuel_tonnes,
                "co2e_tonnes": e.objectives.co2e_tonnes,
                "cost_usd": e.objectives.cost_usd,
                "violation": e.objectives.violation,
                "feasible": e.objectives.violation <= 1e-9,
            }
            for e in archive.entries
        ]
    )


def pareto_front_scatter_3d(archive: ParetoArchive) -> go.Figure:
    df = archive_to_dataframe(archive)
    fig = px.scatter_3d(
        df, x="fuel_tonnes", y="co2e_tonnes", z="cost_usd", color="feasible",
        color_discrete_map={True: "#2E8B57", False: "#B22222"},
        labels={"fuel_tonnes": "Fuel (t)", "co2e_tonnes": "CO2e (t)", "cost_usd": "Cost (USD)"},
        title="Pareto front: fuel vs. emissions vs. cost",
    )
    fig.update_traces(marker=dict(size=5))
    return fig


def convergence_line_chart(histories: dict[str, list[GenerationStats]], metric: str = "hypervolume") -> go.Figure:
    fig = go.Figure()
    for name, history in histories.items():
        fig.add_trace(go.Scatter(x=[h.generation for h in history], y=[getattr(h, metric) for h in history], mode="lines", name=name))
    fig.update_layout(title=f"Convergence: {metric} vs. generation", xaxis_title="Generation", yaxis_title=metric)
    return fig


def fleet_allocation_bar(plan_rows: pd.DataFrame) -> go.Figure:
    """`plan_rows` has columns route_id, vessel_class, fuel_type, count."""
    fig = px.bar(plan_rows, x="route_id", y="count", color="fuel_type", barmode="stack", title="Fleet allocation by route and fuel type")
    return fig


def emissions_by_fuel_bar(plan_rows: pd.DataFrame) -> go.Figure:
    """`plan_rows` needs a co2e_tonnes column alongside fuel_type."""
    grouped = plan_rows.groupby("fuel_type", as_index=False)["co2e_tonnes"].sum()
    fig = px.bar(grouped, x="fuel_type", y="co2e_tonnes", title="Lifecycle CO2e by fuel type", color="fuel_type")
    return fig


def save_static_pareto_front(archive: ParetoArchive, path: str) -> None:
    df = archive_to_dataframe(archive)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    pairs = [("fuel_tonnes", "co2e_tonnes"), ("fuel_tonnes", "cost_usd"), ("co2e_tonnes", "cost_usd")]
    for ax, (x, y) in zip(axes, pairs):
        colors = df["feasible"].map({True: "#2E8B57", False: "#B22222"})
        ax.scatter(df[x], df[y], c=colors, s=25)
        ax.set_xlabel(x)
        ax.set_ylabel(y)
    fig.suptitle("Pareto front projections (green = feasible, red = infeasible)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def save_static_convergence(histories: dict[str, list[GenerationStats]], path: str, metric: str = "hypervolume") -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, history in histories.items():
        ax.plot([h.generation for h in history], [getattr(h, metric) for h in history], label=name)
    ax.set_xlabel("Generation")
    ax.set_ylabel(metric)
    ax.set_title(f"Convergence: {metric} vs. generation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
