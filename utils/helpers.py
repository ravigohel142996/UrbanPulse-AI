"""
utils/helpers.py — Shared utility functions for UrbanPulse AI.
"""

from __future__ import annotations

import time
from functools import wraps
from typing import Any, Callable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Timing decorator
# ---------------------------------------------------------------------------

def timeit(func: Callable) -> Callable:
    """Decorator that prints the execution time of a function."""

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"[timeit] {func.__qualname__} took {elapsed:.3f}s")
        return result

    return wrapper


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def normalise_series(series: pd.Series) -> pd.Series:
    """Min-max normalise a pandas Series to [0, 1]."""
    lo, hi = series.min(), series.max()
    if hi == lo:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - lo) / (hi - lo)


def congestion_ratio(vehicle_count: float, capacity: float) -> float:
    """Return congestion ratio clamped to [0, 1].

    Parameters
    ----------
    vehicle_count:
        Number of vehicles currently on the road segment.
    capacity:
        Maximum vehicles the road can handle per hour.
    """
    if capacity <= 0:
        return 1.0
    return float(np.clip(vehicle_count / capacity, 0.0, 1.0))


def congestion_label(ratio: float) -> str:
    """Return a human-readable congestion label for a given ratio."""
    if ratio < 0.40:
        return "Free Flow"
    if ratio < 0.65:
        return "Moderate"
    if ratio < 0.85:
        return "Heavy"
    return "Severe"


def speed_from_congestion(ratio: float, free_flow_speed: float = 60.0) -> float:
    """Estimate average speed (km/h) from a congestion ratio.

    Uses a simplified BPR (Bureau of Public Roads) function:
    speed = free_flow_speed / (1 + 0.15 * (ratio / (1 - ratio + 1e-6)) ^ 4)
    """
    bpr = 1 + 0.15 * ((ratio / (1 - ratio + 1e-6)) ** 4)
    return float(np.clip(free_flow_speed / bpr, 5.0, free_flow_speed))


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_number(value: float, decimals: int = 1) -> str:
    """Format a large number with K/M suffix."""
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.{decimals}f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.{decimals}f}K"
    return f"{value:.{decimals}f}"


def pct_change_label(current: float, previous: float) -> str:
    """Return a signed percentage-change string (e.g. '+5.2%')."""
    if previous == 0:
        return "N/A"
    pct = (current - previous) / abs(previous) * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"


# ---------------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------------

def safe_sample(df: pd.DataFrame, n: int, random_state: int = 42) -> pd.DataFrame:
    """Return up to *n* rows sampled from *df* without raising an error."""
    return df.sample(min(n, len(df)), random_state=random_state)
