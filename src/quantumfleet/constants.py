"""Shared constants used across data generation, prediction, and optimization."""

NM_TO_KM = 1.852
KNOTS_TO_KM_PER_HOUR = 1.852

DEFAULT_RANDOM_SEED = 42
DEFAULT_PLANNING_PERIOD_DAYS = 30
DEFAULT_BERTH_TIME_DAYS = 1.5

# Naval-architecture rule of thumb: a vessel's lightship (unladen) weight is
# roughly a third of its deadweight tonnage.
LIGHTSHIP_FRACTION_OF_DWT = 0.35
