# -*- coding: utf-8 -*-
"""
Example 23: Automated Feature Engineering
==========================================

Demonstrates FeatureEngineer capabilities:
  1. Log transforms for skewed distributions
  2. Polynomial interactions
  3. Target encoding for high-cardinality categoricals
  4. Date feature extraction
  5. Auto-binning

Requirements:
  pip install scomp-link
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from scomp_link import FeatureEngineer

# --- Generate realistic data ---
np.random.seed(42)
N = 2000

df = pd.DataFrame({
    'salary': np.random.exponential(50000, N),       # skewed
    'experience_years': np.random.exponential(5, N), # skewed
    'age': np.random.normal(38, 10, N),
    'dept': np.random.choice(['eng', 'sales', 'hr', 'marketing', 'finance',
                               'ops', 'legal', 'design', 'data', 'product',
                               'support', 'qa'], N),
    'hire_date': pd.date_range('2015-01-01', periods=N, freq='D'),
    'satisfaction': np.random.uniform(1, 10, N),
})
y = (0.6 * df['salary'] + 5000 * df['experience_years'] +
     200 * df['age'] + np.random.randn(N) * 10000)

print("=" * 60)
print("AUTOMATED FEATURE ENGINEERING")
print("=" * 60)
print(f"Original shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# --- Apply Feature Engineering ---
fe = FeatureEngineer(
    interactions=True,
    log_transform=True,
    skew_threshold=0.8,
    date_features=True,
    target_encode=True,
    target_encode_threshold=8,
    auto_bin=True,
    n_bins=5,
)

X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=0.2, random_state=42)
X_train_eng = fe.fit_transform(X_train, y_train)
X_test_eng = fe.transform(X_test)

print(f"\nEngineered shape: {X_train_eng.shape}")
print(f"New columns added: {X_train_eng.shape[1] - df.shape[1] + 2}")  # +2 for removed cols
print(f"\nAll columns: {list(X_train_eng.columns)}")

# --- Compare model performance ---
print("\n--- Model Comparison ---")

# Without feature engineering (numeric only)
X_raw_num = df.select_dtypes(include=[np.number])
X_tr_raw, X_te_raw, y_tr, y_te = train_test_split(X_raw_num, y, test_size=0.2, random_state=42)
model_raw = LinearRegression()
score_raw = cross_val_score(model_raw, X_tr_raw, y_tr, cv=5, scoring='r2').mean()

# With feature engineering
X_eng_num = X_train_eng.select_dtypes(include=[np.number])
model_eng = LinearRegression()
score_eng = cross_val_score(model_eng, X_eng_num, y_train, cv=5, scoring='r2').mean()

print(f"  Raw features R²:        {score_raw:.4f}")
print(f"  Engineered features R²: {score_eng:.4f}")
print(f"  Improvement:            +{(score_eng - score_raw):.4f}")
