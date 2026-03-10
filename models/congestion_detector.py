"""
models/congestion_detector.py — IsolationForest anomaly detection.

Detects traffic spikes, sudden congestion, and abnormal road usage.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from config import IF_CONTAMINATION, IF_RANDOM_STATE


# ---------------------------------------------------------------------------
# Anomaly feature columns
# ---------------------------------------------------------------------------

ANOMALY_FEATURES: list[str] = [
    "vehicle_count",
    "congestion_ratio",
    "traffic_density",
    "hour",
    "weather_code",
]


# ---------------------------------------------------------------------------
# Detection result
# ---------------------------------------------------------------------------

@dataclass
class AnomalyReport:
    """Summary statistics from one anomaly-detection run."""

    total_records: int
    anomaly_count: int
    anomaly_rate: float
    mean_anomaly_score: float       # higher (closer to 0) = more anomalous
    top_anomalous_roads: list[str]


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class CongestionDetector:
    """IsolationForest wrapper for traffic anomaly detection."""

    def __init__(
        self,
        contamination: float = IF_CONTAMINATION,
        random_state: int = IF_RANDOM_STATE,
    ) -> None:
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self._is_fitted: bool = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "CongestionDetector":
        """Train the detector on historical traffic data.

        Parameters
        ----------
        df:
            Traffic DataFrame (from
            :class:`~data.traffic_generator.TrafficDataGenerator`).

        Returns
        -------
        self (for chaining)
        """
        X = self._prepare(df)
        X_sc = self.scaler.fit_transform(X)
        self.model.fit(X_sc)
        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """Annotate *df* with anomaly flags and scores.

        Adds two columns to a copy of *df*:
        - ``is_anomaly`` (bool)
        - ``anomaly_score`` (float, higher = more anomalous)

        Parameters
        ----------
        df:
            Traffic DataFrame.
        """
        self._check_fitted()
        X = self._prepare(df)
        X_sc = self.scaler.transform(X)

        raw_labels = self.model.predict(X_sc)      # +1 normal, -1 anomaly
        raw_scores = self.model.score_samples(X_sc)  # more negative = more anomalous

        result = df.copy()
        result["is_anomaly"] = raw_labels == -1
        result["anomaly_score"] = -raw_scores      # flip so higher = more anomalous
        return result

    def report(self, df_annotated: pd.DataFrame) -> AnomalyReport:
        """Build a summary report from an annotated DataFrame."""
        anomalies = df_annotated[df_annotated["is_anomaly"]]
        top_roads = (
            anomalies.groupby("road_id")["is_anomaly"]
            .count()
            .nlargest(5)
            .index.tolist()
        )
        return AnomalyReport(
            total_records=len(df_annotated),
            anomaly_count=int(anomalies["is_anomaly"].sum()),
            anomaly_rate=round(len(anomalies) / max(len(df_annotated), 1), 4),
            mean_anomaly_score=float(df_annotated["anomaly_score"].mean()),
            top_anomalous_roads=top_roads,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        available = [c for c in ANOMALY_FEATURES if c in df.columns]
        if len(available) < 2:
            raise ValueError(
                f"Need at least 2 of {ANOMALY_FEATURES}, got: {available}"
            )
        return df[available].fillna(0)

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Call .fit() before .detect().")
