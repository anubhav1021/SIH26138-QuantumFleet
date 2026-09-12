"""Chart builders: matplotlib figures for static report export, plotly
figures for the interactive Streamlit dashboard. Both read directly from a
`ParetoArchive` / `list[GenerationStats]` / benchmark results dict so the
dashboard and the case-study report always show the same data shapes.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from quantumfleet.optimization.pareto import ParetoArchive
from quantumfleet.optimization.problem import Route
from quantumfleet.optimization.qea import GenerationStats
from quantumfleet.reporting.ports import resolve_port_coordinates

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


def actual_vs_predicted_scatter(y_true: np.ndarray, y_pred: np.ndarray, title: str = "Actual vs. predicted fuel consumption") -> go.Figure:
    lo, hi = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_true, y=y_pred, mode="markers", marker=dict(size=5, opacity=0.5), name="Predictions"))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(dash="dash", color="gray"), name="Perfect prediction"))
    fig.update_layout(title=title, xaxis_title="Actual fuel (t)", yaxis_title="Predicted fuel (t)")
    return fig


def residual_scatter(y_true: np.ndarray, y_pred: np.ndarray) -> go.Figure:
    residuals = y_pred - y_true
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_pred, y=residuals, mode="markers", marker=dict(size=5, opacity=0.5)))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(title="Residuals vs. predicted value", xaxis_title="Predicted fuel (t)", yaxis_title="Residual (predicted - actual)")
    return fig


def feature_importance_bar(importances: np.ndarray, names: list[str], top_n: int = 15) -> go.Figure:
    order = np.argsort(importances)[::-1][:top_n]
    fig = px.bar(x=[names[i] for i in order], y=importances[order], labels={"x": "Feature", "y": "Importance"}, title="Feature importance")
    return fig


def _route_endpoint(lat: float | None, lon: float | None, port_name: str | None) -> tuple[float, float] | None:
    if lat is not None and lon is not None:
        return lat, lon
    return resolve_port_coordinates(port_name)


def route_map(routes: list[Route]) -> go.Figure | None:
    """A simple world map with one line per route that has resolvable
    origin/destination coordinates -- either set directly on the Route, or
    resolved by name from the curated port lookup (`reporting.ports`).
    Routes lacking coordinates are omitted, not errored on: most scenarios
    (including every built-in one) don't set these optional fields at all,
    so returning None here is the common case, not a failure. Uses plotly's
    built-in world map (scattergeo) -- no API token or external service."""
    fig = go.Figure()
    plotted_any = False
    for route in routes:
        origin = _route_endpoint(route.origin_lat, route.origin_lon, route.origin_port)
        destination = _route_endpoint(route.destination_lat, route.destination_lon, route.destination_port)
        if origin is None or destination is None:
            continue
        plotted_any = True
        fig.add_trace(
            go.Scattergeo(
                lon=[origin[1], destination[1]],
                lat=[origin[0], destination[0]],
                mode="lines+markers",
                line=dict(width=2),
                marker=dict(size=7),
                name=route.route_id,
                text=[f"{route.route_id}: {route.origin_port or 'origin'}", f"{route.route_id}: {route.destination_port or 'destination'}"],
                hoverinfo="text",
            )
        )
    if not plotted_any:
        return None
    fig.update_geos(showcountries=True, showland=True, landcolor="rgb(235,235,235)", showocean=True, oceancolor="rgb(245,250,255)")
    fig.update_layout(title="Route map", height=500, margin=dict(l=0, r=0, t=40, b=0))
    return fig
