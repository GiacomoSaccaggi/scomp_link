#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example 6: Clustering - Known Number of Categories (KMeans)
Use Case: Unsupervised learning when number of clusters is known
"""

import pandas as pd
import numpy as np
from scomp_link import ScompLinkPipeline

np.random.seed(42)
n_per_cluster = 300
n_clusters = 4
data = []

for i in range(n_clusters):
    center_x = np.random.uniform(-10, 10)
    center_y = np.random.uniform(-10, 10)
    for _ in range(n_per_cluster):
        data.append([
            np.random.normal(center_x, 1.5),
            np.random.normal(center_y, 1.5),
            np.random.normal(0, 1)
        ])

df = pd.DataFrame(data, columns=['feature_1', 'feature_2', 'feature_3'])

print("=" * 70)
print("EXAMPLE 6: Clustering - Known Categories (KMeans)")
print("=" * 70)
print(f"Dataset size: {len(df)} records")
print(f"Expected clusters: {n_clusters}")

pipe = ScompLinkPipeline("Customer Segmentation with Known Clusters")
pipe.set_objectives(["Identify Customer Segments", "Maximize Cluster Separation"])
pipe.import_and_clean_data(df)
df['cluster'] = 0
pipe.select_variables(target_col='cluster')
pipe.choose_model("categorical_unknown", metadata={"categories_known": True, "n_clusters": n_clusters})

results = pipe.run_pipeline(task_type="clustering", n_clusters=n_clusters)

print("\n" + "=" * 70)
print("RESULTS:")
print(f"Model Type: {results['model_type']}")
print(f"Clusters found: {results['n_clusters']}")
print(f"Metrics: {results['metrics']}")
print("=" * 70)

# Note: KMeans clustering assigns existing data to clusters
# For new data prediction, save the model:
print("\n" + "=" * 70)
print("SAVING CLUSTERING MODEL...")
import pickle
import os
os.makedirs('./staging/example_06', exist_ok=True)
with open('./staging/example_06/kmeans.pkl', 'wb') as f:
    from sklearn.cluster import KMeans
    # Re-create model for saving
    kmeans = KMeans(n_clusters=4, random_state=42)
    X = df[['feature_1', 'feature_2', 'feature_3']].values
    kmeans.fit(X)
    pickle.dump(kmeans, f)
print("✅ KMeans model saved to ./staging/example_06/kmeans.pkl")
print("✅ Clustering completed! Use saved model to predict new data clusters.")
print("=" * 70)
