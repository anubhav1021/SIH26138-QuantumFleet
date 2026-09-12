import pytest

from quantumfleet.scenarios.repository import ScenarioRepository, VesselCatalogRepository, slugify

SAMPLE_SCENARIO = {
    "name": "My Test Scenario",
    "carbon_price_usd_per_tonne": 25,
    "green_fuel_pathway": False,
    "routes": [
        {
            "route_id": "R1",
            "distance_nm": 1000,
            "cargo_demand_tonnes": 5000,
            "period_days": 30,
            "max_transit_days": 10,
            "allowed_fuel_types": ["HFO", "MDO"],
            "shore_power_available": True,
            "emission_cap_tonnes_co2e": 5000,
        }
    ],
}


def test_slugify_produces_safe_filenames():
    assert slugify("My Test Scenario!") == "my_test_scenario"
    assert slugify("   ") == "scenario"
    assert slugify("Already_ok-123") == "already_ok_123" or slugify("Already_ok-123") == "already_ok-123"


def test_list_builtin_finds_default_and_demo_scenarios(tmp_path):
    repo = ScenarioRepository(builtin_dir="configs", user_dir=tmp_path)
    keys = {s.key for s in repo.list_builtin()}
    assert keys == {"default_scenario", "demo_case_study"}
    assert all(s.is_builtin for s in repo.list_builtin())


def test_save_load_user_scenario_round_trips(tmp_path):
    repo = ScenarioRepository(builtin_dir="configs", user_dir=tmp_path)
    key = repo.unique_key(SAMPLE_SCENARIO["name"])
    repo.save_user_scenario(key, SAMPLE_SCENARIO)

    loaded = repo.load_raw(key)
    assert loaded["name"] == "My Test Scenario"
    assert len(loaded["routes"]) == 1
    assert not repo.is_builtin(key)

    scenario_config = repo.load_scenario_config(key)
    assert scenario_config.name == "My Test Scenario"
    assert scenario_config.routes[0].route_id == "R1"


def test_user_scenario_appears_in_list_user_not_list_builtin(tmp_path):
    repo = ScenarioRepository(builtin_dir="configs", user_dir=tmp_path)
    key = repo.unique_key(SAMPLE_SCENARIO["name"])
    repo.save_user_scenario(key, SAMPLE_SCENARIO)

    assert key in {s.key for s in repo.list_user()}
    assert key not in {s.key for s in repo.list_builtin()}


def test_cannot_overwrite_builtin_scenario_by_key(tmp_path):
    repo = ScenarioRepository(builtin_dir="configs", user_dir=tmp_path)
    with pytest.raises(ValueError):
        repo.save_user_scenario("default_scenario", SAMPLE_SCENARIO)


def test_duplicate_creates_a_new_independent_key(tmp_path):
    repo = ScenarioRepository(builtin_dir="configs", user_dir=tmp_path)
    new_key = repo.duplicate("default_scenario", "My Copy")
    assert new_key != "default_scenario"
    assert not repo.is_builtin(new_key)

    duplicated = repo.load_raw(new_key)
    original = repo.load_raw("default_scenario")
    assert duplicated["name"] == "My Copy"
    assert len(duplicated["routes"]) == len(original["routes"])


def test_unique_key_avoids_collisions(tmp_path):
    repo = ScenarioRepository(builtin_dir="configs", user_dir=tmp_path)
    key1 = repo.unique_key("Duplicate Name")
    repo.save_user_scenario(key1, {**SAMPLE_SCENARIO, "name": "Duplicate Name"})
    key2 = repo.unique_key("Duplicate Name")
    assert key1 != key2


def test_delete_user_scenario_removes_it(tmp_path):
    repo = ScenarioRepository(builtin_dir="configs", user_dir=tmp_path)
    key = repo.unique_key(SAMPLE_SCENARIO["name"])
    repo.save_user_scenario(key, SAMPLE_SCENARIO)
    repo.delete_user_scenario(key)
    assert key not in {s.key for s in repo.list_user()}
    with pytest.raises(KeyError):
        repo.load_raw(key)


def test_cannot_delete_builtin_scenario(tmp_path):
    repo = ScenarioRepository(builtin_dir="configs", user_dir=tmp_path)
    with pytest.raises(KeyError):
        repo.delete_user_scenario("default_scenario")


def test_persistence_survives_a_fresh_repository_instance(tmp_path):
    """Simulates an application restart: a new ScenarioRepository object
    pointed at the same directory must see what a previous instance saved."""
    repo1 = ScenarioRepository(builtin_dir="configs", user_dir=tmp_path)
    key = repo1.unique_key(SAMPLE_SCENARIO["name"])
    repo1.save_user_scenario(key, SAMPLE_SCENARIO)

    repo2 = ScenarioRepository(builtin_dir="configs", user_dir=tmp_path)
    assert key in {s.key for s in repo2.list_user()}
    assert repo2.load_raw(key)["name"] == "My Test Scenario"


SAMPLE_VESSEL = {
    "id": "CUSTOM_TEST_VESSEL",
    "type": "Container Ship",
    "tier": "Medium",
    "dwt_tonnes": 40000,
    "admiralty_coefficient": 440,
    "day_rate_usd": 10000,
    "min_speed_knots": 12,
    "max_speed_knots": 20,
    "aux_load_kw": 700,
    "compatible_fuels": ["HFO", "MDO", "LNG"],
}


def test_vessel_catalog_load_merged_includes_builtin_and_user(tmp_path):
    user_path = tmp_path / "user_vessels.yaml"
    repo = VesselCatalogRepository(builtin_path="configs/vessel_types.yaml", user_path=user_path)
    repo.save_user_vessel(SAMPLE_VESSEL)

    merged, speed_bins = repo.load_merged()
    assert "CUSTOM_TEST_VESSEL" in merged
    assert "CONTAINER_MEDIUM" in merged  # a known built-in id still present
    assert merged["CUSTOM_TEST_VESSEL"].dwt_tonnes == 40000


def test_vessel_catalog_persists_across_instances(tmp_path):
    user_path = tmp_path / "user_vessels.yaml"
    VesselCatalogRepository(builtin_path="configs/vessel_types.yaml", user_path=user_path).save_user_vessel(SAMPLE_VESSEL)

    repo2 = VesselCatalogRepository(builtin_path="configs/vessel_types.yaml", user_path=user_path)
    merged, _ = repo2.load_merged()
    assert "CUSTOM_TEST_VESSEL" in merged


def test_cannot_save_vessel_with_builtin_id(tmp_path):
    user_path = tmp_path / "user_vessels.yaml"
    repo = VesselCatalogRepository(builtin_path="configs/vessel_types.yaml", user_path=user_path)
    with pytest.raises(ValueError):
        repo.save_user_vessel({**SAMPLE_VESSEL, "id": "CONTAINER_MEDIUM"})


def test_delete_user_vessel(tmp_path):
    user_path = tmp_path / "user_vessels.yaml"
    repo = VesselCatalogRepository(builtin_path="configs/vessel_types.yaml", user_path=user_path)
    repo.save_user_vessel(SAMPLE_VESSEL)
    repo.delete_user_vessel("CUSTOM_TEST_VESSEL")
    merged, _ = repo.load_merged()
    assert "CUSTOM_TEST_VESSEL" not in merged


def test_cannot_delete_builtin_vessel(tmp_path):
    user_path = tmp_path / "user_vessels.yaml"
    repo = VesselCatalogRepository(builtin_path="configs/vessel_types.yaml", user_path=user_path)
    with pytest.raises(KeyError):
        repo.delete_user_vessel("CONTAINER_MEDIUM")


def test_user_vessel_upsert_overwrites_by_id(tmp_path):
    user_path = tmp_path / "user_vessels.yaml"
    repo = VesselCatalogRepository(builtin_path="configs/vessel_types.yaml", user_path=user_path)
    repo.save_user_vessel(SAMPLE_VESSEL)
    repo.save_user_vessel({**SAMPLE_VESSEL, "dwt_tonnes": 99999})

    merged, _ = repo.load_merged()
    assert merged["CUSTOM_TEST_VESSEL"].dwt_tonnes == 99999
    assert len(repo.load_user_raw()) == 1  # upserted, not duplicated
