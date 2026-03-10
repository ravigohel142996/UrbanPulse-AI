"""
config.py — Global configuration constants for UrbanPulse AI.

All tuneable parameters live here so the rest of the codebase never
contains magic numbers.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# City network
# ---------------------------------------------------------------------------
NETWORK_NUM_NODES: int = 30          # intersections
NETWORK_NUM_EDGES: int = 60          # road segments
RANDOM_SEED: int = 42

# Road attributes (ranges used during random generation)
ROAD_LENGTH_MIN: float = 0.5         # km
ROAD_LENGTH_MAX: float = 5.0         # km
TRAFFIC_CAPACITY_MIN: int = 100      # vehicles / hour
TRAFFIC_CAPACITY_MAX: int = 1_000    # vehicles / hour
TRAVEL_TIME_MIN: float = 1.0         # minutes
TRAVEL_TIME_MAX: float = 15.0        # minutes

# ---------------------------------------------------------------------------
# Traffic simulation
# ---------------------------------------------------------------------------
DEFAULT_NUM_VEHICLES: int = 500
DEFAULT_SIMULATION_HOURS: int = 24
RECORDS_PER_HOUR: int = 6            # one record every 10 minutes

WEATHER_CONDITIONS: list[str] = ["clear", "rain", "fog", "snow"]
WEATHER_WEIGHTS: list[float] = [0.60, 0.20, 0.12, 0.08]

# Peak-hour multipliers applied to base traffic volume
PEAK_HOURS: dict[int, float] = {
    7: 1.8, 8: 2.2, 9: 1.6,          # morning rush
    12: 1.4, 13: 1.3,                 # lunch
    17: 2.0, 18: 2.4, 19: 1.7,       # evening rush
}

# ---------------------------------------------------------------------------
# ML models
# ---------------------------------------------------------------------------
RF_N_ESTIMATORS: int = 100
RF_MAX_DEPTH: int = 10
RF_RANDOM_STATE: int = RANDOM_SEED

IF_CONTAMINATION: float = 0.05       # expected anomaly fraction
IF_RANDOM_STATE: int = RANDOM_SEED

TEST_SIZE: float = 0.20
CV_FOLDS: int = 5

# ---------------------------------------------------------------------------
# Route optimisation
# ---------------------------------------------------------------------------
CONGESTION_THRESHOLD: float = 0.75   # ratio: current_traffic / capacity
CONGESTION_PENALTY: float = 5.0      # weight multiplier for congested roads

# ---------------------------------------------------------------------------
# Dashboard / UI
# ---------------------------------------------------------------------------
PAGE_TITLE: str = "UrbanPulse AI — Smart City Traffic Intelligence"
PAGE_ICON: str = "🚦"
LAYOUT: str = "wide"

NORMAL_ROAD_COLOR: str = "#2196F3"   # blue
CONGESTED_ROAD_COLOR: str = "#F44336"  # red
NODE_COLOR: str = "#4CAF50"          # green

CHART_HEIGHT: int = 400
MAP_HEIGHT: int = 500
