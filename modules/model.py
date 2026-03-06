from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error
from xgboost import XGBRegressor


class ModelError(Exception):
    """Raised when model training or prediction cannot run."""


@dataclass
class PredictionResult:
    """Container for prediction outputs used by the Streamlit app."""

    latest_close: float
    linear_prediction: float
    random_forest_prediction: float
    xgboost_prediction: float
    ensemble_prediction: float
    percent_change: float
    confidence: float
    prediction_interval_low: float
    prediction_interval_high: float
    signal: str
    model_errors: Dict[str, float] = field(default_factory=dict)
    metrics_table: pd.DataFrame = field(default_factory=pd.DataFrame)
    test_predictions: pd.DataFrame = field(default_factory=pd.DataFrame)


# This list keeps feature selection consistent across all models.
FEATURE_COLS = ["lag_1", "lag_2", "lag_3", "ma_5", "ma_10"]
BOOTSTRAP_ITERATIONS = 200
BOOTSTRAP_BLOCK_SIZE = 10
CONFIDENCE_SCALING_FACTOR = 25.0


def _build_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create lag and moving-average features from historical returns."""
    df = data.copy()
    close = pd.to_numeric(df["Close"], errors="coerce")

    # Use daily returns as the modeling base series.
    df["return"] = close.pct_change()

    # Build lag-based features from the same return series.
    df["lag_1"] = df["return"].shift(1)
    df["lag_2"] = df["return"].shift(2)
    df["lag_3"] = df["return"].shift(3)

    # Build moving-average features from the same return series.
    df["ma_5"] = df["return"].rolling(window=5).mean()
    df["ma_10"] = df["return"].rolling(window=10).mean()

    # Predict the next-day return.
    df["target"] = df["return"].shift(-1)

    # Keep raw close values to convert returns back to prices later.
    df["close_raw"] = close

    return df


def _train_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> XGBRegressor:
    """Train an XGBoost regressor with beginner-friendly starter settings."""
    xgb_model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=4,
        random_state=42,
        objective="reg:squarederror",
    )
    xgb_model.fit(X_train, y_train)
    return xgb_model


def _moving_block_bootstrap_indices(n_samples: int, block_size: int, rng: np.random.Generator) -> np.ndarray:
    """Sample contiguous index blocks (with replacement) to preserve local time dependence."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")

    block = max(1, min(block_size, n_samples))
    max_start = n_samples - block
    sampled_indices: list[int] = []

    while len(sampled_indices) < n_samples:
        start = int(rng.integers(0, max_start + 1))
        sampled_indices.extend(range(start, start + block))

    return np.array(sampled_indices[:n_samples], dtype=int)


def train_and_predict(data: pd.DataFrame) -> PredictionResult:
    """Train LR/RF/XGBoost, then estimate uncertainty with moving-block bootstrap.

    The point forecast comes from models retrained on all clean samples. For
    uncertainty, we repeatedly resample the training window using contiguous
    blocks (moving-block bootstrap) so adjacent observations keep their local
    time-series structure. Each bootstrap sample retrains LR/RF/XGBoost and
    produces an ensemble next-day return prediction. The 2.5th/97.5th
    percentiles of this bootstrap distribution define a 95% prediction interval.
    """
    # Build all features first, then keep the latest row for tomorrow's forecast.
    full_df = _build_features(data)
    latest_row = full_df.iloc[[-1]]

    if latest_row[FEATURE_COLS].isnull().any(axis=None):
        raise ModelError(
            "Latest data row has missing feature values. "
            "Try a longer history period."
        )

    latest_features = latest_row[FEATURE_COLS]
    latest_close = float(latest_row["close_raw"].iloc[0])

    # Drop rows with missing feature/target values before model training.
    train_df = full_df.dropna(subset=FEATURE_COLS + ["target", "close_raw"]).copy()
    if len(train_df) < 80:
        raise ModelError("Not enough historical data to train the models.")

    X = train_df[FEATURE_COLS]
    y = train_df["target"]

    # Time-series split: use oldest 80% for training and newest 20% for testing.
    # Shuffling is bad for time-series because it leaks future patterns into training.
    split_index = int(len(train_df) * 0.8)
    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    y_train, y_test = y.iloc[:split_index], y.iloc[split_index:]

    # Train evaluation models on the training window only.
    linear_eval = LinearRegression()
    linear_eval.fit(X_train, y_train)

    rf_eval = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        min_samples_leaf=2,
    )
    rf_eval.fit(X_train, y_train)

    xgb_eval = _train_xgboost(X_train, y_train)

    # Predict returns for the held-out test window.
    linear_test_return = linear_eval.predict(X_test)
    rf_test_return = rf_eval.predict(X_test)
    xgb_test_return = xgb_eval.predict(X_test)

    # Convert return predictions to price predictions for easier interpretation.
    test_prev_close = train_df.loc[X_test.index, "close_raw"]
    actual_test_price = test_prev_close * (1 + y_test)
    linear_test_price = test_prev_close * (1 + linear_test_return)
    rf_test_price = test_prev_close * (1 + rf_test_return)
    xgb_test_price = test_prev_close * (1 + xgb_test_return)

    # Calculate requested evaluation metrics on held-out test prices.
    metrics_table = pd.DataFrame(
        {
            "Model": ["Linear Regression", "Random Forest", "XGBoost"],
            "MAE": [
                mean_absolute_error(actual_test_price, linear_test_price),
                mean_absolute_error(actual_test_price, rf_test_price),
                mean_absolute_error(actual_test_price, xgb_test_price),
            ],
            "RMSE": [
                np.sqrt(mean_squared_error(actual_test_price, linear_test_price)),
                np.sqrt(mean_squared_error(actual_test_price, rf_test_price)),
                np.sqrt(mean_squared_error(actual_test_price, xgb_test_price)),
            ],
            "MAPE": [
                mean_absolute_percentage_error(actual_test_price, linear_test_price),
                mean_absolute_percentage_error(actual_test_price, rf_test_price),
                mean_absolute_percentage_error(actual_test_price, xgb_test_price),
            ],
        }
    )

    # Keep a test-period DataFrame for plotting actual vs model predictions in Streamlit.
    test_predictions = pd.DataFrame(
        {
            "Actual": actual_test_price,
            "Linear Regression": linear_test_price,
            "Random Forest": rf_test_price,
            "XGBoost": xgb_test_price,
        },
        index=X_test.index,
    )

    # Retrain models on all available clean data for the final next-day forecast.
    linear_model = LinearRegression()
    linear_model.fit(X, y)

    rf_model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        min_samples_leaf=2,
    )
    rf_model.fit(X, y)

    xgb_model = _train_xgboost(X, y)

    # Predict tomorrow's return, then convert each model result to price.
    linear_return = float(linear_model.predict(latest_features)[0])
    rf_return = float(rf_model.predict(latest_features)[0])
    xgb_return = float(xgb_model.predict(latest_features)[0])
    ensemble_return = (linear_return + rf_return + xgb_return) / 3

    linear_prediction = latest_close * (1 + linear_return)
    random_forest_prediction = latest_close * (1 + rf_return)
    xgboost_prediction = latest_close * (1 + xgb_return)
    ensemble_prediction = latest_close * (1 + ensemble_return)

    percent_change = ensemble_return * 100

    # Moving-block bootstrap uncertainty estimation on the training window.
    rng = np.random.default_rng(42)
    bootstrap_ensemble_returns: list[float] = []

    for _ in range(BOOTSTRAP_ITERATIONS):
        sampled_idx = _moving_block_bootstrap_indices(
            n_samples=len(X_train),
            block_size=BOOTSTRAP_BLOCK_SIZE,
            rng=rng,
        )

        X_boot = X_train.iloc[sampled_idx]
        y_boot = y_train.iloc[sampled_idx]

        linear_boot = LinearRegression()
        linear_boot.fit(X_boot, y_boot)

        rf_boot = RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            min_samples_leaf=2,
        )
        rf_boot.fit(X_boot, y_boot)

        xgb_boot = _train_xgboost(X_boot, y_boot)

        linear_boot_return = float(linear_boot.predict(latest_features)[0])
        rf_boot_return = float(rf_boot.predict(latest_features)[0])
        xgb_boot_return = float(xgb_boot.predict(latest_features)[0])

        bootstrap_ensemble_returns.append(
            (linear_boot_return + rf_boot_return + xgb_boot_return) / 3
        )

    interval_low_return, interval_high_return = np.percentile(
        bootstrap_ensemble_returns,
        [2.5, 97.5],
    )
    interval_width_in_percent = float((interval_high_return - interval_low_return) * 100)
    confidence = float(np.clip(
        100 - (interval_width_in_percent * CONFIDENCE_SCALING_FACTOR),
        0,
        100,
    ))

    prediction_interval_low = latest_close * (1 + float(interval_low_return))
    prediction_interval_high = latest_close * (1 + float(interval_high_return))

    signal = (
        "▲ Bullish (Likely Rise)"
        if ensemble_prediction >= latest_close
        else "▼ Bearish (Likely Fall)"
    )

    return PredictionResult(
        latest_close=latest_close,
        linear_prediction=linear_prediction,
        random_forest_prediction=random_forest_prediction,
        xgboost_prediction=xgboost_prediction,
        ensemble_prediction=ensemble_prediction,
        percent_change=percent_change,
        confidence=confidence,
        prediction_interval_low=prediction_interval_low,
        prediction_interval_high=prediction_interval_high,
        signal=signal,
        model_errors={
            "Linear Regression MAE (test price)": float(metrics_table.loc[0, "MAE"]),
            "Random Forest MAE (test price)": float(metrics_table.loc[1, "MAE"]),
            "XGBoost MAE (test price)": float(metrics_table.loc[2, "MAE"]),
        },
        metrics_table=metrics_table,
        test_predictions=test_predictions,
    )
