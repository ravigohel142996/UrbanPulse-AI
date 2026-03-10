"""
data/traffic_generator.py — Stochastic traffic simulation engine.

Produces a DataFrame of synthetic traffic observations that mirror
real-world patterns (rush hours, weather effects, special events).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from config import (
    DEFAULT_NUM_VEHICLES,
    DEFAULT_SIMULATION_HOURS,
    PEAK_HOURS,
    RANDOM_SEED,
    RECORDS_PER_HOUR,
    WEATHER_CONDITIONS,
    WEATHER_WEIGHTS,
)
from network.road_network import CityRoadNetwork


# ---------------------------------------------------------------------------
# Weather effect multipliers on traffic volume
# ---------------------------------------------------------------------------

WEATHER_EFFECT: dict[str, float] = {
    "clear": 1.00,
    "rain": 1.25,   # more vehicles → slower speeds
    "fog": 1.40,
    "snow": 1.70,
}

# ---------------------------------------------------------------------------
# Simulation core
# ---------------------------------------------------------------------------


class TrafficDataGenerator:
    """Generates synthetic traffic records for a city road network."""

    def __init__(
        self,
        network: CityRoadNetwork,
        seed: int = RANDOM_SEED,
    ) -> None:
        self.network = network
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        num_vehicles: int = DEFAULT_NUM_VEHICLES,
        simulation_hours: int = DEFAULT_SIMULATION_HOURS,
        event_probability: float = 0.05,
    ) -> pd.DataFrame:
        """Run the simulation and return a tidy traffic DataFrame.

        Parameters
        ----------
        num_vehicles:
            Base number of vehicles active at any point in time.
        simulation_hours:
            Duration of the simulation window (hours).
        event_probability:
            Probability that a special event is occurring in any hour.

        Returns
        -------
        pd.DataFrame
            One row per road segment per time step with columns:
            road_id, timestamp, hour, vehicle_count, weather_condition,
            event_indicator, road_type, traffic_capacity, congestion_ratio,
            traffic_density, time_of_day.
        """
        edges = list(self.network.graph.edges(data=True))
        if not edges:
            raise ValueError("Network has no edges – build the graph first.")

        records: list[dict] = []
        time_steps = simulation_hours * RECORDS_PER_HOUR

        for step in range(time_steps):
            hour = (step // RECORDS_PER_HOUR) % 24
            minute = (step % RECORDS_PER_HOUR) * (60 // RECORDS_PER_HOUR)
            timestamp = pd.Timestamp("2024-01-15") + pd.Timedelta(
                hours=hour, minutes=minute
            )
            weather = self._rng.choice(
                WEATHER_CONDITIONS,
                p=WEATHER_WEIGHTS,
            )
            event = 1 if self._rng.random() < event_probability else 0

            peak_factor = PEAK_HOURS.get(hour, 1.0)
            weather_factor = WEATHER_EFFECT.get(weather, 1.0)
            event_factor = 1.5 if event else 1.0
            base_vehicles = num_vehicles * peak_factor * weather_factor * event_factor

            for u, v, data in edges:
                seg = data["segment"]
                # Each road gets a share of the total vehicles (Poisson noise)
                share = self._rng.exponential(scale=1.0 / max(len(edges), 1))
                raw_count = int(base_vehicles * share)
                vehicle_count = max(0, min(raw_count, seg.traffic_capacity * 2))

                congestion = vehicle_count / max(seg.traffic_capacity, 1)
                traffic_density = min(vehicle_count / max(seg.road_length, 0.1), 500)

                records.append(
                    {
                        "road_id": seg.road_id,
                        "timestamp": timestamp,
                        "hour": hour,
                        "vehicle_count": vehicle_count,
                        "weather_condition": weather,
                        "event_indicator": event,
                        "road_type": seg.road_type,
                        "traffic_capacity": seg.traffic_capacity,
                        "congestion_ratio": round(congestion, 4),
                        "traffic_density": round(traffic_density, 2),
                        "time_of_day": self._time_of_day_label(hour),
                    }
                )

        df = pd.DataFrame(records)
        df = self._encode_categoricals(df)
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _time_of_day_label(hour: int) -> str:
        if 5 <= hour < 10:
            return "morning"
        if 10 <= hour < 14:
            return "midday"
        if 14 <= hour < 19:
            return "evening"
        return "night"

    @staticmethod
    def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
        """Add numeric encodings for categorical columns."""
        weather_map = {"clear": 0, "rain": 1, "fog": 2, "snow": 3}
        road_type_map = {"highway": 0, "arterial": 1, "collector": 2, "local": 3}
        time_map = {"morning": 0, "midday": 1, "evening": 2, "night": 3}

        df["weather_code"] = df["weather_condition"].map(weather_map).fillna(0).astype(int)
        df["road_type_code"] = df["road_type"].map(road_type_map).fillna(1).astype(int)
        df["time_code"] = df["time_of_day"].map(time_map).fillna(3).astype(int)
        return df
