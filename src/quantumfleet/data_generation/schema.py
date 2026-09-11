"""Column-name constants for the voyage_records table.

Every downstream module (features.py, ml_model.py, validate.py) imports
these constants instead of hardcoding column-name strings. That is the
concrete mechanism that makes "swap in real AIS/noon-report data later"
actually true: a future real-data loader only needs to produce a dataframe
with these same columns and data_source="real"; nothing downstream changes.
"""


class VoyageRecordColumns:
    RECORD_ID = "record_id"
    VESSEL_ID = "vessel_id"
    VESSEL_CLASS = "vessel_class"
    DWT_TONNES = "dwt_tonnes"
    LIGHTSHIP_TONNES = "lightship_tonnes"
    LOAD_FACTOR = "load_factor"
    DISPLACEMENT_TONNES = "displacement_tonnes"
    ROUTE_ID = "route_id"
    DAY_OF_VOYAGE = "day_of_voyage"
    PLANNED_SPEED_KNOTS = "planned_speed_knots"
    SPEED_KNOTS = "speed_knots"
    HEADING_DEG = "heading_deg"
    WIND_SPEED_KNOTS = "wind_speed_knots"
    WIND_DIR_DEG = "wind_dir_deg"
    WAVE_HEIGHT_M = "wave_height_m"
    SEA_STATE_CATEGORY = "sea_state_category"
    HULL_FOULING_DAYS = "hull_fouling_days"
    FUEL_TYPE = "fuel_type"
    MAIN_ENGINE_POWER_KW = "main_engine_power_kw"
    FUEL_CONSUMED_TONNES = "fuel_consumed_tonnes"
    CO2E_EMITTED_TONNES = "co2e_emitted_tonnes"
    DATA_SOURCE = "data_source"

    ALL = [
        RECORD_ID, VESSEL_ID, VESSEL_CLASS, DWT_TONNES, LIGHTSHIP_TONNES,
        LOAD_FACTOR, DISPLACEMENT_TONNES, ROUTE_ID, DAY_OF_VOYAGE,
        PLANNED_SPEED_KNOTS, SPEED_KNOTS, HEADING_DEG, WIND_SPEED_KNOTS,
        WIND_DIR_DEG, WAVE_HEIGHT_M, SEA_STATE_CATEGORY, HULL_FOULING_DAYS,
        FUEL_TYPE, MAIN_ENGINE_POWER_KW, FUEL_CONSUMED_TONNES,
        CO2E_EMITTED_TONNES, DATA_SOURCE,
    ]

    # Explicit input features named in the PS delivery table: speed, load,
    # weather, vessel type. fuel_type is included because alternative fuels
    # change fuel mass per unit energy (see fuels.properties).
    FEATURE_COLUMNS = [
        VESSEL_CLASS, SPEED_KNOTS, LOAD_FACTOR, WAVE_HEIGHT_M,
        WIND_SPEED_KNOTS, FUEL_TYPE,
    ]

    TARGET_COLUMN = FUEL_CONSUMED_TONNES
    GROUP_COLUMN = VESSEL_ID
