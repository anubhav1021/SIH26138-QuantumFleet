"""Optimization-specific benchmarking metrics not covered by sklearn:
convergence speed and scalability trend across instance sizes."""

import pandas as pd

from quantumfleet.optimization.qea import GenerationStats


def generations_to_reach_hypervolume(history: list[GenerationStats], target_fraction: float = 0.95) -> int | None:
    """First generation at which hypervolume reaches `target_fraction` of the
    run's own final hypervolume. None if it never does (e.g. hypervolume
    stayed at 0 throughout -- no feasible solution found)."""
    if not history:
        return None
    final_hv = history[-1].hypervolume
    if final_hv <= 0:
        return None
    target = final_hv * target_fraction
    for h in history:
        if h.hypervolume >= target:
            return h.generation
    return None


def scalability_table(results: dict[str, dict[int, float]]) -> pd.DataFrame:
    """`results` maps algorithm name -> {n_routes: runtime_seconds}."""
    return pd.DataFrame(results)
