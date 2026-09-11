"""Synthetic voyage-record data generator.

Ground truth per record comes from `prediction.physics_model` (the same
formula used everywhere else in the platform), scaled by a per-vessel fixed
efficiency offset and multiplicative log-normal noise to emulate real
noon-report measurement imprecision. Records are generated at daily
(noon-report) granularity -- one row per vessel-day -- a deliberate realism
choice that also makes a future real-data swap-in credible rather than
aspirational.
"""

import numpy as np
import pandas as pd

from quantumfleet.constants import DEFAULT_RANDOM_SEED, LIGHTSHIP_FRACTION_OF_DWT
from quantumfleet.data_generation.schema import VoyageRecordColumns as C
from quantumfleet.data_generation.vessel_profiles import sample_vessel_fleet
from quantumfleet.data_generation.weather_model import sample_weather, sea_state_category
from quantumfleet.prediction.physics_model import VesselClass, VoyageConditions, predict_fuel_consumption_physics

# Roughly matches today's real fleet fuel mix: conventional fuels dominate,
# alternative fuels are rarer -- gives the ML model realistic class imbalance
# to handle, and vessel classes further restrict this to their compatible list.
FUEL_WEIGHTS_DEFAULT = {
    "HFO": 0.60,
    "MDO": 0.15,
    "LNG": 0.10,
    "METHANOL": 0.05,
    "HYDROGEN": 0.05,
    "AMMONIA": 0.05,
}


def _sample_fuel_type(compatible_fuels: tuple[str, ...], rng: np.random.Generator) -> str:
    weights = np.array([FUEL_WEIGHTS_DEFAULT.get(f, 0.01) for f in compatible_fuels], dtype=float)
    weights /= weights.sum()
    return str(rng.choice(compatible_fuels, p=weights))


def generate_voyage_records(
    vessel_classes: dict[str, VesselClass],
    n_vessels: int = 200,
    legs_per_vessel: int = 20,
    n_historical_routes: int = 15,
    seed: int = DEFAULT_RANDOM_SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    fleet = sample_vessel_fleet(vessel_classes, n_vessels, rng)
    historical_route_ids = [f"HIST-{k:03d}" for k in range(n_historical_routes)]

    n_records = n_vessels * legs_per_vessel
    weather = sample_weather(n_records, rng)

    rows: list[dict] = []
    idx = 0
    for _, vessel_row in fleet.iterrows():
        vessel_class = vessel_classes[vessel_row["vessel_class"]]
        for day in range(legs_per_vessel):
            load_factor = float(np.clip(rng.beta(5, 2), 0.2, 1.0))
            speed_knots = float(max(1.0, rng.normal(vessel_row["planned_speed_knots"], 0.75)))
            wave_height_m = float(weather["wave_height_m"][idx])
            headwind_knots = float(weather["headwind_knots"][idx])
            fuel_type = _sample_fuel_type(vessel_class.compatible_fuels, rng)
            distance_nm = speed_knots * 24.0  # a full noon-report day's steaming

            conditions = VoyageConditions(
                speed_knots=speed_knots,
                load_factor=load_factor,
                distance_nm=distance_nm,
                wave_height_m=wave_height_m,
                headwind_knots=headwind_knots,
            )
            pred = predict_fuel_consumption_physics(vessel_class, conditions, fuel_type)

            noise = float(rng.lognormal(0, 0.06))
            noisy_fuel_tonnes = pred.fuel_tonnes * vessel_row["efficiency_offset"] * noise
            noisy_co2e_tonnes = pred.co2e_tonnes * vessel_row["efficiency_offset"] * noise

            rows.append(
                {
                    C.RECORD_ID: f"R{idx:06d}",
                    C.VESSEL_ID: vessel_row["vessel_id"],
                    C.VESSEL_CLASS: vessel_row["vessel_class"],
                    C.DWT_TONNES: vessel_class.dwt_tonnes,
                    C.LIGHTSHIP_TONNES: LIGHTSHIP_FRACTION_OF_DWT * vessel_class.dwt_tonnes,
                    C.LOAD_FACTOR: load_factor,
                    C.DISPLACEMENT_TONNES: LIGHTSHIP_FRACTION_OF_DWT * vessel_class.dwt_tonnes + vessel_class.dwt_tonnes * load_factor,
                    C.ROUTE_ID: str(rng.choice(historical_route_ids)),
                    C.DAY_OF_VOYAGE: day,
                    C.PLANNED_SPEED_KNOTS: vessel_row["planned_speed_knots"],
                    C.SPEED_KNOTS: speed_knots,
                    C.HEADING_DEG: float(weather["heading_deg"][idx]),
                    C.WIND_SPEED_KNOTS: float(weather["wind_speed_knots"][idx]),
                    C.WIND_DIR_DEG: float(weather["wind_dir_deg"][idx]),
                    C.WAVE_HEIGHT_M: wave_height_m,
                    C.SEA_STATE_CATEGORY: sea_state_category(wave_height_m),
                    C.HULL_FOULING_DAYS: 0.0,
                    C.FUEL_TYPE: fuel_type,
                    C.MAIN_ENGINE_POWER_KW: pred.power_kw,
                    C.FUEL_CONSUMED_TONNES: noisy_fuel_tonnes,
                    C.CO2E_EMITTED_TONNES: noisy_co2e_tonnes,
                    C.DATA_SOURCE: "synthetic",
                }
            )
            idx += 1

    return pd.DataFrame(rows, columns=C.ALL)
