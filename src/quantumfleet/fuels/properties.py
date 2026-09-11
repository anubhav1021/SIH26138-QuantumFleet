"""Fuel property constants and conversion functions.

All figures below are representative/approximate values assembled from
standard maritime-engineering and IMO/EU well-to-wake GHG literature orders
of magnitude. They are chosen to be internally consistent -- SFOC (specific
fuel oil consumption) is *derived* from each fuel's lower heating value (LHV)
and an assumed brake thermal efficiency (BTE) rather than independently
guessed, via `SFOC = 3600 / (BTE * LHV)` -- so the platform behaves sensibly
end-to-end. They are not citation-pinned values and should be replaced with
vetted primary sources before any real operational or regulatory
decision-making use.

Shore power is deliberately NOT included in FUEL_LIBRARY: it is grid
electricity that displaces a vessel's at-berth auxiliary/"hotel" load, not a
propulsion fuel choice, so it is modelled as a separate per-assignment flag
(see `optimization.encoding`) rather than a 7th propulsion option here.
"""

from dataclasses import dataclass

SECONDS_PER_HOUR = 3600.0


@dataclass(frozen=True)
class FuelProperties:
    name: str
    lhv_mj_per_kg: float
    density_kg_per_m3: float
    brake_thermal_efficiency: float
    co2e_g_per_mj_grey: float
    co2e_g_per_mj_green: float
    price_usd_per_tonne: float
    price_usd_per_tonne_green: float

    @property
    def sfoc_g_per_kwh(self) -> float:
        return SECONDS_PER_HOUR / (self.brake_thermal_efficiency * self.lhv_mj_per_kg)


FUEL_LIBRARY: dict[str, FuelProperties] = {
    "HFO": FuelProperties(
        name="HFO", lhv_mj_per_kg=40.0, density_kg_per_m3=980.0,
        brake_thermal_efficiency=0.47,
        co2e_g_per_mj_grey=92.0, co2e_g_per_mj_green=92.0,
        price_usd_per_tonne=550.0, price_usd_per_tonne_green=550.0,
    ),
    "MDO": FuelProperties(
        name="MDO", lhv_mj_per_kg=42.7, density_kg_per_m3=860.0,
        brake_thermal_efficiency=0.47,
        co2e_g_per_mj_grey=93.0, co2e_g_per_mj_green=93.0,
        price_usd_per_tonne=750.0, price_usd_per_tonne_green=750.0,
    ),
    "LNG": FuelProperties(
        name="LNG", lhv_mj_per_kg=49.0, density_kg_per_m3=450.0,
        brake_thermal_efficiency=0.48,
        co2e_g_per_mj_grey=80.0, co2e_g_per_mj_green=80.0,
        price_usd_per_tonne=650.0, price_usd_per_tonne_green=650.0,
    ),
    "METHANOL": FuelProperties(
        name="METHANOL", lhv_mj_per_kg=19.9, density_kg_per_m3=792.0,
        brake_thermal_efficiency=0.45,
        co2e_g_per_mj_grey=90.0, co2e_g_per_mj_green=15.0,
        price_usd_per_tonne=500.0, price_usd_per_tonne_green=900.0,
    ),
    "HYDROGEN": FuelProperties(
        name="HYDROGEN", lhv_mj_per_kg=120.0, density_kg_per_m3=71.0,
        brake_thermal_efficiency=0.45,
        co2e_g_per_mj_grey=110.0, co2e_g_per_mj_green=10.0,
        price_usd_per_tonne=3500.0, price_usd_per_tonne_green=6500.0,
    ),
    "AMMONIA": FuelProperties(
        name="AMMONIA", lhv_mj_per_kg=18.6, density_kg_per_m3=682.0,
        brake_thermal_efficiency=0.43,
        co2e_g_per_mj_grey=130.0, co2e_g_per_mj_green=10.0,
        price_usd_per_tonne=400.0, price_usd_per_tonne_green=750.0,
    ),
}

PROPULSION_FUELS: tuple[str, ...] = tuple(FUEL_LIBRARY.keys())

SHORE_POWER_GRID_CO2E_G_PER_KWH_DEFAULT = 550.0
SHORE_POWER_GRID_CO2E_G_PER_KWH_CLEAN = 100.0
SHORE_POWER_PRICE_USD_PER_KWH = 0.18

# Unburned methane escaping a dual-fuel LNG engine has ~28x CO2's GWP100, so
# it materially affects LNG's effective well-to-wake footprint. The grey
# factor above already assumes a nominal slip level; this multiplier lets
# scenario analysis dial slip up/down without touching FUEL_LIBRARY.
LNG_METHANE_SLIP_FACTOR_DEFAULT = 1.0


def energy_to_fuel_mass_kg(energy_mj: float, fuel_type: str) -> float:
    """Required fuel mass (kg) to deliver `energy_mj` of propulsion energy."""
    return energy_mj / FUEL_LIBRARY[fuel_type].lhv_mj_per_kg


def power_and_hours_to_fuel_mass_kg(power_kw: float, duration_hours: float, fuel_type: str) -> float:
    """Fuel mass (kg) burned sustaining `power_kw` for `duration_hours`, via SFOC."""
    fuel = FUEL_LIBRARY[fuel_type]
    return power_kw * duration_hours * fuel.sfoc_g_per_kwh / 1000.0


def fuel_mass_to_volume_m3(fuel_mass_kg: float, fuel_type: str) -> float:
    return fuel_mass_kg / FUEL_LIBRARY[fuel_type].density_kg_per_m3


def fuel_mass_to_co2e_kg(
    fuel_mass_kg: float,
    fuel_type: str,
    *,
    green: bool = False,
    methane_slip_factor: float = LNG_METHANE_SLIP_FACTOR_DEFAULT,
) -> float:
    """Well-to-wake CO2e (kg) released burning `fuel_mass_kg` of `fuel_type`."""
    fuel = FUEL_LIBRARY[fuel_type]
    energy_mj = fuel_mass_kg * fuel.lhv_mj_per_kg
    co2e_g_per_mj = fuel.co2e_g_per_mj_green if green else fuel.co2e_g_per_mj_grey
    if fuel_type == "LNG":
        co2e_g_per_mj *= methane_slip_factor
    return energy_mj * co2e_g_per_mj / 1000.0


def fuel_mass_to_cost_usd(fuel_mass_kg: float, fuel_type: str, *, green: bool = False) -> float:
    fuel = FUEL_LIBRARY[fuel_type]
    price = fuel.price_usd_per_tonne_green if green else fuel.price_usd_per_tonne
    return (fuel_mass_kg / 1000.0) * price


def shore_power_co2e_kg(
    energy_kwh: float, *, grid_co2e_g_per_kwh: float = SHORE_POWER_GRID_CO2E_G_PER_KWH_DEFAULT
) -> float:
    return energy_kwh * grid_co2e_g_per_kwh / 1000.0


def shore_power_cost_usd(energy_kwh: float, *, price_usd_per_kwh: float = SHORE_POWER_PRICE_USD_PER_KWH) -> float:
    return energy_kwh * price_usd_per_kwh
