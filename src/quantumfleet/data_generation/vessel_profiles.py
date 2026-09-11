"""Samples a fixed synthetic fleet of vessels once per generation run."""

import numpy as np
import pandas as pd

from quantumfleet.prediction.physics_model import VesselClass


def sample_vessel_fleet(vessel_classes: dict[str, VesselClass], n_vessels: int, rng: np.random.Generator) -> pd.DataFrame:
    """One row per synthetic vessel: a fixed class, a fixed manufacturing/
    maintenance efficiency offset (~Normal(1.0, 0.03)), and a fixed planned
    cruising speed, all held constant for that vessel's lifetime in the
    dataset -- real fleets have per-vessel idiosyncrasies too."""
    class_ids = list(vessel_classes.keys())
    rows = []
    for i in range(n_vessels):
        vessel_class_id = str(rng.choice(class_ids))
        vessel_class = vessel_classes[vessel_class_id]
        efficiency_offset = float(np.clip(rng.normal(1.0, 0.03), 0.85, 1.15))
        planned_speed = float(rng.uniform(vessel_class.min_speed_knots, vessel_class.max_speed_knots))
        rows.append(
            {
                "vessel_id": f"V{i:04d}",
                "vessel_class": vessel_class_id,
                "efficiency_offset": efficiency_offset,
                "planned_speed_knots": planned_speed,
            }
        )
    return pd.DataFrame(rows)
