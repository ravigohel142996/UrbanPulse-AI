"""
analytics/traffic_metrics.py — KPI and aggregate metric calculations.

All metric functions are pure (no side-effects) so they can be tested
in isolation and reused across the dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from config import CONGESTION_THRESHOLD
from utils.helpers import congestion_ratio, speed_from_congestion


# ---------------------------------------------------------------------------
# KPI snapshot
# ---------------------------------------------------------------------------

@dataclass
class TrafficKPIs:
    """Top-level KPIs displayed in the dashboard overview."""

    total_vehicles: int
    congested_roads: int
    total_roads: int
    average_speed_kmh: float
    traffic_flow_index: float     # 0 = gridlock, 100 = free flow
    anomaly_rate_pct: float


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_kpis(
    df: pd.DataFrame,
    network_edge_count: int,
) -> TrafficKPIs:
    """Compute dashboard KPIs from a traffic DataFrame snapshot.

    Parameters
    ----------
    df:
        Traffic DataFrame (one or more time steps).
    network_edge_count:
        Total number of road segments in the network.
    """
    if df.empty:
        return TrafficKPIs(0, 0, network_edge_count, 0.0, 0.0, 0.0)

    latest = _latest_snapshot(df)

    total_vehicles = int(latest["vehicle_count"].sum())
    congested_roads = int(
        (latest["congestion_ratio"] >= CONGESTION_THRESHOLD).sum()
    )
    avg_ratio = float(latest["congestion_ratio"].mean())
    avg_speed = speed_from_congestion(avg_ratio)
    flow_index = round((1 - avg_ratio) * 100, 1)

    anomaly_col = "is_anomaly"
    if anomaly_col in df.columns:
        anomaly_rate = float(df[anomaly_col].mean() * 100)
    else:
        anomaly_rate = 0.0

    return TrafficKPIs(
        total_vehicles=total_vehicles,
        congested_roads=congested_roads,
        total_roads=network_edge_count,
        average_speed_kmh=round(avg_speed, 1),
        traffic_flow_index=max(0.0, flow_index),
        anomaly_rate_pct=round(anomaly_rate, 2),
    )


def hourly_traffic_volume(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate total vehicle count by hour across all roads.

    Returns a DataFrame with columns: hour, vehicle_count.
    """
    return (
        df.groupby("hour")["vehicle_count"]
        .sum()
        .reset_index()
        .rename(columns={"vehicle_count": "total_vehicles"})
        .sort_values("hour")
    )


def congestion_by_road_type(df: pd.DataFrame) -> pd.DataFrame:
    """Mean congestion ratio grouped by road type.

    Returns a DataFrame with columns: road_type, mean_congestion.
    """
    return (
        df.groupby("road_type")["congestion_ratio"]
        .mean()
        .reset_index()
        .rename(columns={"congestion_ratio": "mean_congestion"})
        .sort_values("mean_congestion", ascending=False)
    )


def peak_hour_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Return per-hour stats: mean vehicles, mean congestion, peak flag."""
    hourly = (
        df.groupby("hour")
        .agg(
            mean_vehicles=("vehicle_count", "mean"),
            mean_congestion=("congestion_ratio", "mean"),
            std_vehicles=("vehicle_count", "std"),
        )
        .reset_index()
    )
    threshold = hourly["mean_vehicles"].quantile(0.75)
    hourly["is_peak"] = hourly["mean_vehicles"] >= threshold
    return hourly.sort_values("hour")


def top_congested_roads(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """Return the *n* roads with the highest mean congestion ratio."""
    return (
        df.groupby("road_id")["congestion_ratio"]
        .mean()
        .nlargest(n)
        .reset_index()
        .rename(columns={"congestion_ratio": "mean_congestion"})
    )


def weather_impact_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Mean vehicle count and congestion ratio by weather condition."""
    return (
        df.groupby("weather_condition")
        .agg(
            mean_vehicles=("vehicle_count", "mean"),
            mean_congestion=("congestion_ratio", "mean"),
            record_count=("vehicle_count", "count"),
        )
        .reset_index()
    )


def build_congestion_heatmap_data(df: pd.DataFrame) -> pd.DataFrame:
    """Pivot table: rows = road_id, columns = hour, values = congestion_ratio.

    Suitable for rendering as a heatmap.
    """
    snapshot = (
        df.groupby(["road_id", "hour"])["congestion_ratio"]
        .mean()
        .reset_index()
    )
    pivot = snapshot.pivot(index="road_id", columns="hour", values="congestion_ratio")
    return pivot.fillna(0)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _latest_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    """Return rows from the most recent timestamp in the DataFrame."""
    if "timestamp" not in df.columns:
        return df
    latest_ts = df["timestamp"].max()
    return df[df["timestamp"] == latest_ts]
