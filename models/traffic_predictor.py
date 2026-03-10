"""
models/traffic_predictor.py — RandomForest-based traffic flow prediction.

Trains on historical simulation data and predicts future vehicle counts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler

from config import (
    CV_FOLDS,
    RF_MAX_DEPTH,
    RF_N_ESTIMATORS,
    RF_RANDOM_STATE,
    TEST_SIZE,
)


# ---------------------------------------------------------------------------
# Feature / target column names
# ---------------------------------------------------------------------------

FEATURE_COLS: list[str] = [
    "hour",
    "weather_code",
    "road_type_code",
    "traffic_density",
    "event_indicator",
    "time_code",
    "traffic_capacity",
]
TARGET_COL: str = "vehicle_count"


# ---------------------------------------------------------------------------
# Evaluation result
# ---------------------------------------------------------------------------

@dataclass
class ModelEvaluation:
    """Holds train/test evaluation metrics for the predictor."""

    mae: float
    rmse: float
    r2: float
    cv_r2_mean: float
    cv_r2_std: float
    feature_importances: dict[str, float] = field(default_factory=dict)

    def __str__(self) -> str:
        return (
            f"MAE={self.mae:.2f}, RMSE={self.rmse:.2f}, "
            f"R²={self.r2:.4f}, CV-R²={self.cv_r2_mean:.4f}±{self.cv_r2_std:.4f}"
        )


# ---------------------------------------------------------------------------
# Predictor
# ---------------------------------------------------------------------------

class TrafficPredictor:
    """Wraps a RandomForestRegressor with fit/predict/evaluate helpers."""

    def __init__(
        self,
        n_estimators: int = RF_N_ESTIMATORS,
        max_depth: int = RF_MAX_DEPTH,
        random_state: int = RF_RANDOM_STATE,
    ) -> None:
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self._is_fitted: bool = False
        self.evaluation: ModelEvaluation | None = None
        self._X_test: np.ndarray | None = None
        self._y_test: np.ndarray | None = None
        self._y_pred: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "TrafficPredictor":
        """Fit the model on a traffic DataFrame.

        Parameters
        ----------
        df:
            DataFrame produced by :class:`~data.traffic_generator.TrafficDataGenerator`.

        Returns
        -------
        self (for chaining)
        """
        X, y = self._prepare(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RF_RANDOM_STATE
        )

        X_train_sc = self.scaler.fit_transform(X_train)
        X_test_sc = self.scaler.transform(X_test)

        self.model.fit(X_train_sc, y_train)
        self._is_fitted = True

        # Evaluate
        y_pred = self.model.predict(X_test_sc)
        cv_scores = cross_val_score(
            self.model, X_train_sc, y_train, cv=CV_FOLDS, scoring="r2", n_jobs=-1
        )

        self.evaluation = ModelEvaluation(
            mae=float(mean_absolute_error(y_test, y_pred)),
            rmse=float(np.sqrt(mean_squared_error(y_test, y_pred))),
            r2=float(r2_score(y_test, y_pred)),
            cv_r2_mean=float(cv_scores.mean()),
            cv_r2_std=float(cv_scores.std()),
            feature_importances=dict(
                zip(FEATURE_COLS, self.model.feature_importances_.tolist())
            ),
        )

        # Store test data for visualisation
        self._X_test = X_test_sc
        self._y_test = y_test
        self._y_pred = y_pred

        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict vehicle counts for new observations.

        Parameters
        ----------
        df:
            DataFrame with at least the feature columns.

        Returns
        -------
        np.ndarray of predicted vehicle counts.
        """
        self._check_fitted()
        X, _ = self._prepare(df, require_target=False)
        X_sc = self.scaler.transform(X)
        return self.model.predict(X_sc)

    def predict_with_actuals(
        self, df: pd.DataFrame
    ) -> pd.DataFrame:
        """Return a DataFrame with both actual and predicted values."""
        self._check_fitted()
        preds = self.predict(df)
        result = df[["road_id", "timestamp", "hour", TARGET_COL]].copy()
        result["predicted_traffic_flow"] = preds.astype(int)
        result["residual"] = result[TARGET_COL] - result["predicted_traffic_flow"]
        return result

    # ------------------------------------------------------------------
    # Accessors for test data (used in charts)
    # ------------------------------------------------------------------

    @property
    def test_actuals(self) -> np.ndarray | None:
        return self._y_test

    @property
    def test_predictions(self) -> np.ndarray | None:
        return self._y_pred

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare(
        self,
        df: pd.DataFrame,
        require_target: bool = True,
    ) -> tuple[pd.DataFrame, pd.Series | None]:
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns: {missing}")

        X = df[FEATURE_COLS].copy()
        y = df[TARGET_COL].copy() if require_target and TARGET_COL in df.columns else None
        return X, y

    def _check_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Call .fit() before .predict().")
