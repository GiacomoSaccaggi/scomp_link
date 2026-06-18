# -*- coding: utf-8 -*-
"""
Example 22: Pipeline Persistence (.scomp format)
=================================================

Demonstrates saving and loading complete ML pipelines:
  1. Save model + preprocessor + config + metrics as .scomp
  2. Load and predict in one call
  3. Feature schema validation
  4. Sample data for drift detection

Requirements:
  pip install scomp-link
"""

import numpy as np
import pandas as pd
import tempfile
import os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from scomp_link import ScompArtifact

# --- Generate synthetic data ---
np.random.seed(42)
N = 1000
X = pd.DataFrame({
    'temperature': np.random.normal(25, 5, N),
    'humidity': np.random.normal(60, 15, N),
    'pressure': np.random.normal(1013, 10, N),
    'wind_speed': np.random.exponential(5, N),
})
y = 0.5 * X['temperature'] - 0.3 * X['humidity'] + 0.1 * X['pressure'] + np.random.randn(N) * 2

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train model with preprocessing ---
print("=" * 60)
print("PIPELINE PERSISTENCE (.scomp format)")
print("=" * 60)

preprocessor = StandardScaler()
X_train_scaled = preprocessor.fit_transform(X_train)

model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
model.fit(X_train_scaled, y_train)

X_test_scaled = preprocessor.transform(X_test)
r2 = model.score(X_test_scaled, y_test)
print(f"Model R²: {r2:.4f}")

# === SAVE as .scomp ===
print("\n--- Saving pipeline artifact ---")
artifact = ScompArtifact()
artifact.set_model(model)
artifact.set_preprocessor(preprocessor)
artifact.set_config(
    task_type='regression',
    target_col='energy_output',
    feature_cols=['temperature', 'humidity', 'pressure', 'wind_speed'],
    model_params={'n_estimators': 100, 'max_depth': 4},
)
artifact.set_metrics({'r2': round(r2, 4), 'rmse': 1.98, 'mae': 1.55})
artifact.set_feature_schema(X_train)
artifact.set_sample_data(X_train, max_rows=200)
artifact.set_metadata(
    author='data_science_team',
    description='Energy output prediction from weather features',
    experiment_id='exp_042',
    version='1.0.0',
)

save_path = tempfile.mktemp(suffix='.scomp')
artifact.save(save_path)

# === LOAD and predict ===
print("\n--- Loading and predicting ---")
loaded = ScompArtifact.load(save_path)

# predict() chains preprocessor + model automatically
predictions = loaded.predict(X_test)
print(f"Predictions shape: {predictions.shape}")
print(f"Prediction range: [{predictions.min():.2f}, {predictions.max():.2f}]")

# === INSPECT artifact ===
print("\n--- Artifact info ---")
info = loaded.info()
for key, val in info.items():
    print(f"  {key}: {val}")

# === FEATURE SCHEMA ===
print("\n--- Feature schema ---")
for feat, schema in loaded.feature_schema.items():
    if 'min' in schema:
        print(f"  {feat}: [{schema['min']:.2f}, {schema['max']:.2f}] (mean={schema['mean']:.2f})")

# === SAMPLE DATA for drift detection ===
print(f"\n--- Sample data: {loaded.sample_data.shape} ---")
print(loaded.sample_data.head(3).to_string())

# === Verify .scomp file ===
print(f"\n--- File validation ---")
print(f"Is valid .scomp: {ScompArtifact.is_scomp_file(save_path)}")
print(f"File size: {os.path.getsize(save_path) / 1024:.1f} KB")

# Cleanup
os.unlink(save_path)
print("\n✅ Demo complete — artifact saved, loaded, and verified.")
