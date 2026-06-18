# -*- coding: utf-8 -*-
"""
Example 24: Time Series Forecasting
====================================

Demonstrates TimeSeriesForecaster:
  1. ARIMA on trend data
  2. Exponential Smoothing on seasonal data
  3. Walk-forward cross-validation
  4. Confidence intervals

Requirements:
  pip install scomp-link statsmodels
"""

import numpy as np
import pandas as pd
from scomp_link import TimeSeriesForecaster

np.random.seed(42)

# --- Generate synthetic time series ---
t = np.arange(200)
trend = 50 + 0.3 * t
seasonality = 15 * np.sin(2 * np.pi * t / 12)
noise = np.random.randn(200) * 3
series = pd.Series(trend + seasonality + noise)

print("=" * 60)
print("TIME SERIES FORECASTING")
print("=" * 60)
print(f"Series length: {len(series)}, range: [{series.min():.1f}, {series.max():.1f}]")

# === 1. ARIMA ===
print("\n--- 1. ARIMA Forecasting ---")
fc_arima = TimeSeriesForecaster(method='arima', horizon=20)
fc_arima.fit(series[:180])
pred_arima = fc_arima.predict()
actual = series[180:200].values
mae_arima = np.abs(actual - pred_arima.values).mean()
print(f"  Forecast MAE: {mae_arima:.2f}")

# === 2. Exponential Smoothing ===
print("\n--- 2. Exponential Smoothing ---")
fc_ets = TimeSeriesForecaster(method='exp_smoothing', horizon=20, seasonal_period=12)
fc_ets.fit(series[:180])
pred_ets = fc_ets.predict()
mae_ets = np.abs(actual - pred_ets.values).mean()
print(f"  Forecast MAE: {mae_ets:.2f}")

# === 3. Confidence Intervals ===
print("\n--- 3. Forecast with Confidence Intervals ---")
ci = fc_arima.predict_with_ci(steps=10, alpha=0.05)
print(ci.head().to_string())

# === 4. Walk-Forward CV ===
print("\n--- 4. Walk-Forward Cross-Validation ---")
fc_cv = TimeSeriesForecaster(method='arima', horizon=12)
cv_results = fc_cv.walk_forward_cv(series, n_splits=5, horizon=12)
print(f"  Mean MAE:  {cv_results['mean_mae']:.2f}")
print(f"  Mean RMSE: {cv_results['mean_rmse']:.2f}")
print(f"  Mean MAPE: {cv_results['mean_mape']:.1f}%")

# === 5. Auto Method Selection ===
print("\n--- 5. Auto Method Selection ---")
fc_auto = TimeSeriesForecaster(method='auto', horizon=10)
fc_auto.fit(series[:180])
pred_auto = fc_auto.predict()
print(f"  Auto-selected method, forecast mean: {pred_auto.mean():.2f}")

# === Comparison ===
print("\n" + "=" * 60)
print("COMPARISON (MAE on last 20 points)")
print(f"  ARIMA:              {mae_arima:.2f}")
print(f"  Exp. Smoothing:     {mae_ets:.2f}")
print(f"  Best: {'ETS' if mae_ets < mae_arima else 'ARIMA'}")
