# `voyage_records` schema

One row = one vessel-day (noon-report granularity). Column names are defined once in
[`src/quantumfleet/data_generation/schema.py`](../../src/quantumfleet/data_generation/schema.py)
(`VoyageRecordColumns`) and imported everywhere else — nothing downstream hardcodes a
column-name string. A real AIS/noon-report data source can be swapped in by producing a
dataframe with exactly these columns and `data_source="real"`; no downstream code changes.

| Column | Type | Units / values | Notes |
|---|---|---|---|
| `record_id` | str | — | unique per row |
| `vessel_id` | str | — | stable per vessel across its rows; use for group-aware train/test splits |
| `vessel_class` | str | one of the 15 IDs in `configs/vessel_types.yaml` | encodes vessel type + size tier |
| `dwt_tonnes` | float | deadweight tonnage | fixed per vessel_class |
| `lightship_tonnes` | float | tonnes | ≈ 0.35 × DWT |
| `load_factor` | float | 0–1 | fraction of DWT actually carried this leg |
| `displacement_tonnes` | float | tonnes | lightship + DWT × load_factor |
| `route_id` | str | — | historical grouping only; **not** a model feature, and unrelated to the optimizer's scenario route IDs |
| `day_of_voyage` | int | — | 0-indexed day within that vessel's synthetic voyage |
| `planned_speed_knots` | float | knots | fixed per vessel |
| `speed_knots` | float | knots | actual speed this leg — **feature** |
| `heading_deg` | float | 0–360 | — |
| `wind_speed_knots` | float | knots | — **feature** |
| `wind_dir_deg` | float | 0–360 | — |
| `wave_height_m` | float | metres, significant wave height | — **feature** |
| `sea_state_category` | str | calm / slight / moderate / rough / very_rough | derived from `wave_height_m`, for human-readable display only |
| `hull_fouling_days` | float | days since last hull clean | currently always 0 (cut from MVP scope, see plan) |
| `fuel_type` | str | HFO / MDO / LNG / METHANOL / HYDROGEN / AMMONIA | — **feature** |
| `main_engine_power_kw` | float | kW | synthetic-only diagnostic; not used as a model input (it's downstream of speed/load/weather, so using it as a feature would leak the answer) |
| `fuel_consumed_tonnes` | float | tonnes | **target** |
| `co2e_emitted_tonnes` | float | tonnes CO2-equivalent, well-to-wake | derived from `fuel_consumed_tonnes`, not an independent measurement |
| `data_source` | str | `"synthetic"` or `"real"` | provenance flag |

**Features used by the prediction model** (`prediction/features.py`): `vessel_class`,
`speed_knots`, `load_factor`, `wave_height_m`, `wind_speed_knots`, `fuel_type` — matching the
PS delivery table's explicit input-feature list (speed, load, weather, vessel type), plus
`fuel_type` since alternative fuels change fuel mass per unit of propulsion energy.

**Train/test split**: by `vessel_id` (`GroupShuffleSplit`), not by row — each vessel carries a
fixed efficiency offset (see `data_generation/vessel_profiles.py`), so splitting by row would
leak that vessel-specific offset across the split and overstate accuracy.

## `routes` (scenario input, not synthetic-generated)

Routes are not randomly generated — they're defined directly in scenario YAML configs
(`configs/default_scenario.yaml`, `configs/demo_case_study.yaml`) since they represent a
given fleet-planning scenario to optimize, not historical training data:

| Field | Type | Notes |
|---|---|---|
| `route_id` | str | unique per route |
| `distance_nm` | float | one-way distance |
| `cargo_demand_tonnes` | float | per planning period |
| `period_days` | float | planning horizon |
| `max_transit_days` | float | schedule-reliability constraint |
| `allowed_fuel_types` | list[str] | subset of the 6 propulsion fuels available at this route's ports |
| `shore_power_available` | bool | whether destination berths offer a shore-power hookup |
| `emission_cap_tonnes_co2e` | float | regulatory/voluntary emissions ceiling for this route |
