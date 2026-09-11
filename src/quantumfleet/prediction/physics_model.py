"""Physics-based (Admiralty-formula-style) fuel consumption baseline.

Used both as the ground-truth generator for synthetic training data
(`data_generation.generator`) and as the deterministic "physics-only"
comparison point in benchmarking. The optimizer's fitness function
(`optimization.qea`) also calls this directly rather than the trained ML
model, keeping the search loop fast and deterministic and keeping the
prediction and optimization deliverables independently correct.

The Admiralty-coefficient method is a long-standing, historically
empirical/calibrated naval-architecture approximation relating power to
displacement and speed -- not a first-principles hydrodynamic computation --
appropriate for a fast, comparative estimate rather than an exact one.
"""

from dataclasses import dataclass

from quantumfleet.constants import LIGHTSHIP_FRACTION_OF_DWT
from quantumfleet.fuels import properties as fp
from quantumfleet.utils.io import load_yaml


@dataclass(frozen=True)
class VesselClass:
    id: str
    type: str
    tier: str
    dwt_tonnes: float
    admiralty_coefficient: float
    day_rate_usd: float
    min_speed_knots: float
    max_speed_knots: float
    aux_load_kw: float
    compatible_fuels: tuple[str, ...]


@dataclass(frozen=True)
class VoyageConditions:
    speed_knots: float
    load_factor: float
    distance_nm: float
    wave_height_m: float = 0.0
    headwind_knots: float = 0.0
    fouling_days: float = 0.0


@dataclass(frozen=True)
class FuelPrediction:
    power_kw: float
    duration_hours: float
    fuel_tonnes: float
    co2e_tonnes: float


def displacement_tonnes(vessel: VesselClass, load_factor: float) -> float:
    lightship = LIGHTSHIP_FRACTION_OF_DWT * vessel.dwt_tonnes
    return lightship + vessel.dwt_tonnes * load_factor


def admiralty_power_kw(displacement_tonnes: float, speed_knots: float, admiralty_coefficient: float) -> float:
    if speed_knots <= 0 or displacement_tonnes <= 0:
        return 0.0
    return (displacement_tonnes ** (2.0 / 3.0) * speed_knots**3) / admiralty_coefficient


def resistance_multiplier(wave_height_m: float = 0.0, headwind_knots: float = 0.0, fouling_days: float = 0.0) -> float:
    """Empirical added-resistance approximation: Hs=2m -> 1.16x, Hs=4m -> 1.64x,
    Hs=6m -> 2.44x, consistent with typical added-resistance-in-waves orders
    of magnitude reported in ship-performance literature (a smooth
    approximation, not a seakeeping computation)."""
    return 1.0 + 0.04 * wave_height_m**2 + 0.01 * max(0.0, headwind_knots) + 0.0003 * fouling_days


def predict_fuel_consumption_physics(vessel: VesselClass, conditions: VoyageConditions, fuel_type: str) -> FuelPrediction:
    disp = displacement_tonnes(vessel, conditions.load_factor)
    base_power_kw = admiralty_power_kw(disp, conditions.speed_knots, vessel.admiralty_coefficient)
    multiplier = resistance_multiplier(conditions.wave_height_m, conditions.headwind_knots, conditions.fouling_days)
    effective_power_kw = base_power_kw * multiplier

    duration_hours = conditions.distance_nm / conditions.speed_knots if conditions.speed_knots > 0 else 0.0

    fuel_kg = fp.power_and_hours_to_fuel_mass_kg(effective_power_kw, duration_hours, fuel_type)
    co2e_kg = fp.fuel_mass_to_co2e_kg(fuel_kg, fuel_type)

    return FuelPrediction(
        power_kw=effective_power_kw,
        duration_hours=duration_hours,
        fuel_tonnes=fuel_kg / 1000.0,
        co2e_tonnes=co2e_kg / 1000.0,
    )


def auxiliary_fuel_consumption_at_berth(
    vessel: VesselClass,
    berth_hours: float,
    fuel_type: str,
    use_shore_power: bool,
    *,
    grid_co2e_g_per_kwh: float = fp.SHORE_POWER_GRID_CO2E_G_PER_KWH_DEFAULT,
) -> FuelPrediction:
    """At-berth hotel-load energy: either burned onboard as `fuel_type`, or
    supplied by shore power (grid electricity, zero onboard fuel burn)."""
    energy_kwh = vessel.aux_load_kw * berth_hours
    if use_shore_power:
        co2e_kg = fp.shore_power_co2e_kg(energy_kwh, grid_co2e_g_per_kwh=grid_co2e_g_per_kwh)
        return FuelPrediction(power_kw=vessel.aux_load_kw, duration_hours=berth_hours, fuel_tonnes=0.0, co2e_tonnes=co2e_kg / 1000.0)

    fuel_kg = fp.power_and_hours_to_fuel_mass_kg(vessel.aux_load_kw, berth_hours, fuel_type)
    co2e_kg = fp.fuel_mass_to_co2e_kg(fuel_kg, fuel_type)
    return FuelPrediction(power_kw=vessel.aux_load_kw, duration_hours=berth_hours, fuel_tonnes=fuel_kg / 1000.0, co2e_tonnes=co2e_kg / 1000.0)


def load_vessel_catalog(path: str) -> tuple[dict[str, VesselClass], list[float]]:
    """Load the vessel-class catalog and global speed bins from a YAML config
    (see configs/vessel_types.yaml). Returns (vessel_classes_by_id, speed_bins_knots)."""
    raw = load_yaml(path)
    vessel_classes = {
        entry["id"]: VesselClass(
            id=entry["id"],
            type=entry["type"],
            tier=entry["tier"],
            dwt_tonnes=float(entry["dwt_tonnes"]),
            admiralty_coefficient=float(entry["admiralty_coefficient"]),
            day_rate_usd=float(entry["day_rate_usd"]),
            min_speed_knots=float(entry["min_speed_knots"]),
            max_speed_knots=float(entry["max_speed_knots"]),
            aux_load_kw=float(entry["aux_load_kw"]),
            compatible_fuels=tuple(entry["compatible_fuels"]),
        )
        for entry in raw["vessel_classes"]
    }
    speed_bins = [float(s) for s in raw["speed_bins_knots"]]
    return vessel_classes, speed_bins
