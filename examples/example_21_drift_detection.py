# -*- coding: utf-8 -*-
"""
Example 21: Data Drift Detection
=================================

Demonstrates DriftDetector for monitoring production data quality:
  1. PSI (Population Stability Index) per feature
  2. KS 2-sample test
  3. Drift visualization
  4. Feature distribution comparison

Requirements:
  pip install scomp-link
"""

import numpy as np
import pandas as pd
from scomp_link import DriftDetector

# --- Simulate reference (training) data ---
np.random.seed(42)
N_REF = 5000

reference = pd.DataFrame({
    'age': np.random.normal(35, 10, N_REF),
    'income': np.random.lognormal(10.5, 0.5, N_REF),
    'credit_score': np.random.normal(700, 50, N_REF),
    'num_transactions': np.random.poisson(30, N_REF).astype(float),
    'account_balance': np.random.normal(5000, 2000, N_REF),
})

print("=" * 60)
print("DATA DRIFT DETECTION")
print("=" * 60)
print(f"Reference data shape: {reference.shape}")

# === Scenario 1: No Drift ===
print("\n--- Scenario 1: No Drift (same distribution) ---")
current_ok = pd.DataFrame({
    'age': np.random.normal(35, 10, 1000),
    'income': np.random.lognormal(10.5, 0.5, 1000),
    'credit_score': np.random.normal(700, 50, 1000),
    'num_transactions': np.random.poisson(30, 1000).astype(float),
    'account_balance': np.random.normal(5000, 2000, 1000),
})

detector = DriftDetector(reference, psi_threshold=0.2, ks_alpha=0.05)
report_ok = detector.detect(current_ok)
summary_ok = detector.summary(report_ok)
print(f"  Drifted features: {summary_ok['drifted_features']}/{summary_ok['total_features']}")
print(f"  Max PSI: {summary_ok['max_psi']:.4f}")

# === Scenario 2: Gradual Drift ===
print("\n--- Scenario 2: Gradual Drift (slight shift in income + age) ---")
current_drift = pd.DataFrame({
    'age': np.random.normal(38, 12, 1000),       # shifted mean +3, wider std
    'income': np.random.lognormal(10.8, 0.6, 1000),  # shifted
    'credit_score': np.random.normal(700, 50, 1000),  # unchanged
    'num_transactions': np.random.poisson(30, 1000).astype(float),  # unchanged
    'account_balance': np.random.normal(5000, 2000, 1000),  # unchanged
})

report_drift = detector.detect(current_drift)
summary_drift = detector.summary(report_drift)
print(f"  Drifted features: {summary_drift['drifted_features']}/{summary_drift['total_features']}")
print(f"  Worst feature: {summary_drift['worst_feature']} (PSI={summary_drift['max_psi']:.4f})")
print(f"\n  Full report:")
print(report_drift[['feature', 'psi', 'ks_statistic', 'drifted']].to_string(index=False))

# === Scenario 3: Catastrophic Drift ===
print("\n--- Scenario 3: Catastrophic Drift (completely different population) ---")
current_bad = pd.DataFrame({
    'age': np.random.normal(55, 5, 1000),          # elderly population
    'income': np.random.lognormal(11.5, 0.3, 1000),  # much higher income
    'credit_score': np.random.normal(800, 20, 1000),  # all high credit
    'num_transactions': np.random.poisson(80, 1000).astype(float),  # very active
    'account_balance': np.random.normal(20000, 5000, 1000),  # high balance
})

report_bad = detector.detect(current_bad)
summary_bad = detector.summary(report_bad)
print(f"  Drifted features: {summary_bad['drifted_features']}/{summary_bad['total_features']}")
print(f"  Drift %: {summary_bad['drift_pct']:.0f}%")
print(f"\n  PSI values:")
for _, row in report_bad.iterrows():
    status = "🔴 DRIFT" if row['drifted'] else "🟢 OK"
    print(f"    {row['feature']:20s} PSI={row['psi']:.4f}  {status}")

# === Visualization ===
print("\n--- Generating plots ---")
fig_report = detector.plot_drift_report(report_bad)
print(f"✅ Drift bar chart generated")

fig_dist = detector.plot_feature_distribution('income', current_bad)
print(f"✅ Distribution overlay for 'income' generated")

# === Custom feature subset ===
print("\n--- Selective detection (only financial features) ---")
report_selective = detector.detect(current_bad, features=['income', 'credit_score', 'account_balance'])
print(f"  Checked {len(report_selective)} features, {report_selective['drifted'].sum()} drifted")
