#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example 7: Clustering - Unknown Number of Categories (Mean-Shift)
Use Case: Unsupervised learning when number of clusters is unknown
"""

import pandas as pd
import numpy as np
from scomp_link import ScompLinkPipeline

np.random.seed(42)
data = []

for _ in range(5):
    n_points = np.random.randint(100, 300)
    center_x = np.random.uniform(-20, 20)
    center_y = np.random.uniform(-20, 20)
    for _ in range(n_points):
        data.append([
            np.random.normal(center_x, 2),
            np.random.normal(center_y, 2)
        ])

df = pd.DataFrame(data, columns=['feature_1', 'feature_2'])

print("=" * 70)
print("EXAMPLE 7: Clustering - Unknown Categories (Mean-Shift)")
print("=" * 70)
print(f"Dataset size: {len(df)} records")

pipe = ScompLinkPipeline("Anomaly Detection with Unknown Clusters")
pipe.set_objectives(["Discover Natural Groupings", "No Prior Cluster Knowledge"])
pipe.import_and_clean_data(df)
df['cluster'] = 0
pipe.select_variables(target_col='cluster')
pipe.choose_model("categorical_unknown", metadata={"categories_known": False})

results = pipe.run_pipeline(task_type="clustering")

print("\n" + "=" * 70)
print("RESULTS:")
print(f"Model Type: {results['model_type']}")
print(f"Clusters discovered: {results['n_clusters']}")
print(f"Metrics: {results['metrics']}")
print("=" * 70)

# Note: Mean-Shift discovers clusters automatically
# For new data prediction, save the model:
print("\n" + "=" * 70)
print("SAVING CLUSTERING MODEL...")
import pickle
import os
os.makedirs('./staging/example_07', exist_ok=True)
with open('./staging/example_07/meanshift.pkl', 'wb') as f:
    from sklearn.cluster import MeanShift, estimate_bandwidth
    # Re-create model for saving
    X = df[['feature_1', 'feature_2']].values
    bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=min(500, len(X)))
    ms = MeanShift(bandwidth=bandwidth)
    ms.fit(X)
    pickle.dump(ms, f)
print("✅ Mean-Shift model saved to ./staging/example_07/meanshift.pkl")
print("✅ Clustering completed! Use saved model to predict new data clusters.")
print("=" * 70)
