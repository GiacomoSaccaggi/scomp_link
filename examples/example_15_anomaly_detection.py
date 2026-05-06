# -*- coding: utf-8 -*-
"""
Example 15: Anomaly Detection (Multi-Method Consensus)
======================================================

Demonstrates the AnomalyDetector class with 4 methods:
  1. Isolation Forest (tree-based)
  2. Local Outlier Factor (density-based)
  3. TabNet Autoencoder (neural attention)
  4. Transformer Autoencoder (self-attention)

Requirements:
  pip install scomp-link pytorch-tabnet torch
"""

import numpy as np
import pandas as pd
from scomp_link import AnomalyDetector

# --- Generate synthetic data with known anomalies ---
np.random.seed(42)
N = 5000
N_ANOMALIES = 50

# Normal data: 3 correlated features
normal = np.random.randn(N - N_ANOMALIES, 3) * [10, 50, 5] + [20, 100, 10]

# Anomalies: extreme values in different dimensions
anomalies = np.random.randn(N_ANOMALIES, 3) * [50, 200, 30] + [100, 500, 50]

data = np.vstack([normal, anomalies])
df = pd.DataFrame(data, columns=["frequency", "duration", "panelists"])

# --- Run AnomalyDetector ---
detector = AnomalyDetector(
    contamination=0.02,
    methods=["iforest", "lof", "tabnet", "transformer"],
    tabnet_epochs=30,
    transformer_epochs=50,
    consensus_threshold=2,
)

results = detector.fit_predict(df, features=["frequency", "duration", "panelists"])

# --- Results ---
print("\n=== Detector Comparison ===")
print(results["comparison"].to_string(index=False))

# Check how many true anomalies were caught
true_labels = np.array([0] * (N - N_ANOMALIES) + [1] * N_ANOMALIES)
detected = results["data"]["is_anomaly"].values
tp = (detected & true_labels.astype(bool)).sum()
print(f"\nTrue Positives: {tp}/{N_ANOMALIES}")
print(f"False Positives: {detected.sum() - tp}")

# --- Using only specific methods ---
print("\n=== Only Isolation Forest + LOF (no deep learning) ===")
detector_fast = AnomalyDetector(
    contamination=0.02,
    methods=["iforest", "lof"],
    consensus_threshold=2,
)
results_fast = detector_fast.fit_predict(df, features=["frequency", "duration", "panelists"])
print(results_fast["comparison"].to_string(index=False))
