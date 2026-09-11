"""Correlated per-record weather sampling for the synthetic voyage generator."""

import numpy as np


def sample_weather(n: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Wave height ~ Weibull(shape=1.6) scaled to a ~1.8m mean sea state; wind
    speed correlated with wave height plus noise; heading/wind direction
    uniform; headwind is the component of wind opposing the vessel's heading
    (following wind is not penalized, matching physics_model.resistance_multiplier)."""
    wave_height_m = rng.weibull(1.6, n) * 1.8
    wind_speed_knots = np.clip(6.0 * wave_height_m + rng.normal(0, 2.0, n), 0, None)
    wind_dir_deg = rng.uniform(0, 360, n)
    heading_deg = rng.uniform(0, 360, n)
    headwind_knots = wind_speed_knots * np.cos(np.radians(wind_dir_deg - heading_deg))
    return {
        "wave_height_m": wave_height_m,
        "wind_speed_knots": wind_speed_knots,
        "wind_dir_deg": wind_dir_deg,
        "heading_deg": heading_deg,
        "headwind_knots": headwind_knots,
    }


def sea_state_category(wave_height_m: float) -> str:
    if wave_height_m < 0.5:
        return "calm"
    if wave_height_m < 1.25:
        return "slight"
    if wave_height_m < 2.5:
        return "moderate"
    if wave_height_m < 4.0:
        return "rough"
    return "very_rough"
