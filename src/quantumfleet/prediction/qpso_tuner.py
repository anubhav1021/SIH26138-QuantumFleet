"""Quantum-behaved Particle Swarm Optimization (QPSO), per Sun, Feng & Xu
(2004), used to tune the Random Forest prediction model's hyperparameters.

Unlike classical PSO, QPSO particles carry no velocity: each particle's next
position is sampled from a distribution centered between its personal best
and the swarm's global best, contracting around the swarm's mean-best
position (mbest) as the search progresses via the shrinking beta
(contraction-expansion) coefficient. This is what makes the fuel-consumption
*prediction* deliverable quantum-inspired in its own right, independent of
the fleet-optimization QEA engine in `optimization.qea`.
"""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score


@dataclass(frozen=True)
class HyperparamBounds:
    name: str
    low: float
    high: float
    integer: bool = True


DEFAULT_RF_BOUNDS: list[HyperparamBounds] = [
    HyperparamBounds("n_estimators", 20, 300, integer=True),
    HyperparamBounds("max_depth", 3, 30, integer=True),
    HyperparamBounds("min_samples_leaf", 1, 10, integer=True),
    HyperparamBounds("max_features", 0.2, 1.0, integer=False),
]


@dataclass
class QPSOResult:
    best_params: dict
    best_score: float
    history: list[float] = field(default_factory=list)  # global-best-EVER score per iteration


def _decode(position: np.ndarray, bounds: list[HyperparamBounds]) -> dict:
    params = {}
    for value, b in zip(position, bounds):
        v = float(np.clip(value, b.low, b.high))
        params[b.name] = int(round(v)) if b.integer else v
    return params


def rf_cv_rmse_objective(X: np.ndarray, y: np.ndarray, cv: int = 3) -> Callable[[dict], float]:
    """Builds a QPSO objective: cross-validated RMSE (lower is better) of a
    RandomForestRegressor configured with the candidate hyperparameters."""

    def objective(params: dict) -> float:
        model = RandomForestRegressor(random_state=42, **params)
        scores = cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error")
        return float(-scores.mean())

    return objective


def quantum_pso_tune(
    objective: Callable[[dict], float],
    bounds: list[HyperparamBounds] = DEFAULT_RF_BOUNDS,
    n_particles: int = 12,
    n_iterations: int = 15,
    seed: int = 42,
    beta_start: float = 1.0,
    beta_end: float = 0.5,
) -> QPSOResult:
    """Minimizes `objective` (a dict-of-hyperparams -> score callable) over
    the given bounds. `history` tracks the global-best-EVER score (not the
    current swarm best), so it is monotonically non-increasing by
    construction -- the right series to plot for a convergence curve."""
    rng = np.random.default_rng(seed)
    dim = len(bounds)
    lows = np.array([b.low for b in bounds])
    highs = np.array([b.high for b in bounds])

    positions = rng.uniform(lows, highs, size=(n_particles, dim))
    pbest_positions = positions.copy()
    pbest_scores = np.array([objective(_decode(p, bounds)) for p in positions])

    gbest_idx = int(np.argmin(pbest_scores))
    gbest_position = pbest_positions[gbest_idx].copy()
    gbest_score = float(pbest_scores[gbest_idx])
    history = [gbest_score]

    for t in range(n_iterations):
        beta = beta_start - (beta_start - beta_end) * (t / max(1, n_iterations - 1))
        mbest = pbest_positions.mean(axis=0)

        for i in range(n_particles):
            phi = rng.uniform(0, 1, size=dim)
            attractor = phi * pbest_positions[i] + (1 - phi) * gbest_position
            u = rng.uniform(1e-6, 1.0, size=dim)
            sign = rng.choice([-1.0, 1.0], size=dim)
            positions[i] = attractor + sign * beta * np.abs(mbest - positions[i]) * np.log(1.0 / u)
            positions[i] = np.clip(positions[i], lows, highs)

            score = objective(_decode(positions[i], bounds))
            if score < pbest_scores[i]:
                pbest_scores[i] = score
                pbest_positions[i] = positions[i].copy()
                if score < gbest_score:
                    gbest_score = score
                    gbest_position = positions[i].copy()

        history.append(gbest_score)

    return QPSOResult(best_params=_decode(gbest_position, bounds), best_score=gbest_score, history=history)
