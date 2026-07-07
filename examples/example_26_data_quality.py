# -*- coding: utf-8 -*-
"""
Example 26: Data Quality Report
================================

Demonstrates DataQualityReport:
  1. Missing value analysis
  2. Cardinality profiling
  3. Constant feature detection
  4. Duplicate detection
  5. High correlation detection
  6. HTML report generation

Requirements:
  pip install scomp-link
"""

import os
import tempfile

import numpy as np
import pandas as pd

from scomp_link import DataQualityReport

np.random.seed(42)
N = 5000

# --- Create a messy dataset ---
df = pd.DataFrame(
    {
        "user_id": range(N),
        "age": np.random.normal(35, 10, N),
        "income": np.random.lognormal(10.5, 0.5, N),
        "credit_score": np.random.normal(700, 50, N),
        "credit_score_v2": None,  # will be highly correlated
        "transactions": np.random.poisson(20, N).astype(float),
        "category": np.random.choice(["A", "B", "C"], N),
        "status": "active",  # constant
        "email": [f"user_{i}@example.com" for i in range(N)],  # near-unique
        "notes": np.where(np.random.rand(N) > 0.15, np.nan, "some text"),  # 85% missing
        "score_pct": np.random.rand(N) * 100,
    }
)

# High correlation: credit_score_v2 ≈ credit_score
df["credit_score_v2"] = df["credit_score"] + np.random.randn(N) * 0.5

# Add duplicates
df = pd.concat([df, df.sample(100, random_state=42)], ignore_index=True)

print("=" * 60)
print("DATA QUALITY REPORT")
print("=" * 60)
print(f"Dataset shape: {df.shape}")

# --- Generate report ---
dqr = DataQualityReport(df)
report = dqr.generate()

# --- Display results ---
print("\n--- Overview ---")
for k, v in report["overview"].items():
    print(f"  {k}: {v}")

print("\n--- Missing Values ---")
if len(report["missing"]) > 0:
    print(report["missing"].to_string(index=False))
else:
    print("  No missing values")

print("\n--- Constants ---")
print(f"  {report['constants']}")

print("\n--- Duplicates ---")
print(f"  {report['duplicates']['n_duplicates']} duplicates ({report['duplicates']['duplicate_pct']:.1f}%)")

print("\n--- High Correlations (≥0.95) ---")
if len(report["correlations"]) > 0:
    print(report["correlations"].to_string(index=False))
else:
    print("  None found")

print("\n--- Cardinality Flags ---")
flagged = report["cardinality"][report["cardinality"]["flag"] != ""]
print(flagged[["column", "n_unique", "flag"]].to_string(index=False))

# --- HTML Report ---
print("\n--- Generating HTML Report ---")
fd, path = tempfile.mkstemp(suffix=".html")
os.close(fd)
dqr.save_html(path)
print(f"  File size: {os.path.getsize(path) / 1024:.1f} KB")
os.unlink(path)
