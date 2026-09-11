import math

from quantumfleet.fuels import properties as fp


def test_all_fuels_have_positive_properties():
    for fuel in fp.FUEL_LIBRARY.values():
        assert fuel.lhv_mj_per_kg > 0
        assert fuel.density_kg_per_m3 > 0
        assert 0 < fuel.brake_thermal_efficiency < 1
        assert fuel.sfoc_g_per_kwh > 0


def test_sfoc_internally_consistent_with_lhv_and_bte():
    for fuel in fp.FUEL_LIBRARY.values():
        implied_bte = fp.SECONDS_PER_HOUR / (fuel.sfoc_g_per_kwh * fuel.lhv_mj_per_kg)
        assert math.isclose(implied_bte, fuel.brake_thermal_efficiency, rel_tol=1e-9)


def test_energy_mass_round_trip():
    for name in fp.PROPULSION_FUELS:
        mass_kg = fp.energy_to_fuel_mass_kg(1000.0, name)
        fuel = fp.FUEL_LIBRARY[name]
        assert math.isclose(mass_kg * fuel.lhv_mj_per_kg, 1000.0, rel_tol=1e-9)


def test_power_hours_to_mass_matches_sfoc_definition():
    mass_kg = fp.power_and_hours_to_fuel_mass_kg(1000.0, 10.0, "HFO")
    fuel = fp.FUEL_LIBRARY["HFO"]
    expected = 1000.0 * 10.0 * fuel.sfoc_g_per_kwh / 1000.0
    assert math.isclose(mass_kg, expected, rel_tol=1e-9)


def test_mass_to_volume_uses_density():
    vol = fp.fuel_mass_to_volume_m3(1000.0, "HYDROGEN")
    assert math.isclose(vol, 1000.0 / fp.FUEL_LIBRARY["HYDROGEN"].density_kg_per_m3, rel_tol=1e-9)


def test_hydrogen_needs_less_mass_but_more_volume_than_hfo_for_same_energy():
    energy_mj = 100_000.0
    hfo_mass = fp.energy_to_fuel_mass_kg(energy_mj, "HFO")
    h2_mass = fp.energy_to_fuel_mass_kg(energy_mj, "HYDROGEN")
    assert h2_mass < hfo_mass

    hfo_vol = fp.fuel_mass_to_volume_m3(hfo_mass, "HFO")
    h2_vol = fp.fuel_mass_to_volume_m3(h2_mass, "HYDROGEN")
    assert h2_vol > hfo_vol


def test_methanol_and_ammonia_need_more_mass_than_hfo_for_same_energy():
    energy_mj = 100_000.0
    hfo_mass = fp.energy_to_fuel_mass_kg(energy_mj, "HFO")
    for alt in ("METHANOL", "AMMONIA"):
        assert fp.energy_to_fuel_mass_kg(energy_mj, alt) > hfo_mass


def test_green_pathway_emissions_lower_than_grey_for_alt_fuels():
    for name in ("METHANOL", "HYDROGEN", "AMMONIA"):
        grey = fp.fuel_mass_to_co2e_kg(1000.0, name, green=False)
        green = fp.fuel_mass_to_co2e_kg(1000.0, name, green=True)
        assert green < grey


def test_lng_methane_slip_increases_emissions():
    base = fp.fuel_mass_to_co2e_kg(1000.0, "LNG", methane_slip_factor=1.0)
    higher_slip = fp.fuel_mass_to_co2e_kg(1000.0, "LNG", methane_slip_factor=1.5)
    assert higher_slip > base


def test_shore_power_helpers():
    assert fp.shore_power_co2e_kg(100.0) > 0
    assert fp.shore_power_cost_usd(100.0) == 100.0 * fp.SHORE_POWER_PRICE_USD_PER_KWH
