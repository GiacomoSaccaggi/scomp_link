# -*- coding: utf-8 -*-
"""
Example 16: Time Series Anomaly Detection
==========================================

Demonstrates TimeSeriesAnomalyDetector with multiple methods:
  1. Conv1D Autoencoder (deep learning)
  2. Moving Average (statistical)
  3. Moving Median with MAD (robust statistical)
  4. ARIMA Residuals (model-based)

Requirements:
  pip install scomp-link statsmodels
  pip install tensorflow  # only for autoencoder method
"""

import numpy as np
from scomp_link import TimeSeriesAnomalyDetector

# --- Generate synthetic time series ---
np.random.seed(42)
N = 1000

# Normal pattern: sinusoidal + noise
t = np.linspace(0, 20 * np.pi, N)
normal_data = 20 + 5 * np.sin(t) + np.random.randn(N) * 0.5

# Test data: same pattern but with injected anomalies
test_data = normal_data.copy()
test_data[200:210] = 50   # sustained spike
test_data[500:505] = -10  # sustained dip
test_data[800] = 100      # single point anomaly

# --- Method 1: Statistical methods only (fast, no deep learning) ---
print("=== Statistical Methods ===")
detector_stat = TimeSeriesAnomalyDetector(
    methods=['moving_avg', 'moving_median', 'arima'],
    window_size=30,
    n_sigma=3.0,
    arima_order=(5, 1, 0),
)
detector_stat.fit(normal_data)
results_stat = detector_stat.detect(test_data)

print(f"\nResults:")
for method, flags in results_stat['methods'].items():
    print(f"  {method}: {flags.sum()} anomalies")
print(f"  Consensus (any method): {results_stat['anomalies'].sum()} anomalies")

# --- Method 2: With Autoencoder (requires tensorflow) ---
try:
    print("\n=== With Conv1D Autoencoder ===")
    detector_full = TimeSeriesAnomalyDetector(
        methods=['autoencoder', 'moving_avg', 'moving_median'],
        time_steps=50,  # shorter for demo
        ae_epochs=20,
        window_size=30,
        n_sigma=3.0,
        threshold_percentile=95.0,
    )
    detector_full.fit(normal_data)
    results_full = detector_full.detect(test_data)

    print(f"\nResults:")
    for method, flags in results_full['methods'].items():
        print(f"  {method}: {flags.sum()} anomalies")
    print(f"  Consensus (any method): {results_full['anomalies'].sum()} anomalies")
except ImportError:
    print("  TensorFlow not installed, skipping autoencoder method.")
    print("  Install with: pip install tensorflow")
