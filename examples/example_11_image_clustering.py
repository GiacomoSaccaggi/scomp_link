#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example 11: Image Clustering - Unsupervised Learning
Use Case: Group similar images without labels
"""

import pandas as pd
import numpy as np
from scomp_link import ScompLinkPipeline

print("=" * 70)
print("EXAMPLE 11: Image Clustering - Unsupervised Learning")
print("=" * 70)

try:
    np.random.seed(42)
    n_samples = 400
    img_size = 32 * 32 * 3
    
    X_images = []
    true_clusters = []
    
    for cluster_id in range(4):
        for _ in range(n_samples // 4):
            base_value = cluster_id * 0.25
            img = np.random.normal(base_value, 0.05, img_size)
            X_images.append(img)
            true_clusters.append(cluster_id)
    
    df = pd.DataFrame({'image_data': X_images, 'true_cluster': true_clusters})
    
    print(f"Dataset size: {len(df)} records")
    print(f"True clusters: {df['true_cluster'].nunique()}")
    
    pipe = ScompLinkPipeline("Image Clustering")
    pipe.set_objectives(["Discover Image Groups", "Unsupervised Learning"])
    pipe.import_and_clean_data(df)
    df['cluster'] = 0
    pipe.select_variables(target_col='cluster')
    pipe.choose_model("categorical_unknown", metadata={"categories_known": True, "n_clusters": 4})
    
    results = pipe.run_pipeline(task_type="image_clustering", image_col='image_data', n_clusters=4)
    
    print("\n" + "=" * 70)
    print("RESULTS:")
    print(f"Model Type: {results['model_type']}")
    print(f"Clusters found: {results['n_clusters']}")
    print(f"Metrics: {results['metrics']}")
    print("=" * 70)
    
    print("\n✅ Image clustering completed!")
    
    # Save model
    print("\n" + "=" * 70)
    print("SAVING MODEL...")
    model_path = pipe.save_model('./staging/example_11')
    print("=" * 70)
    
    # Load model and predict on new images
    print("\n" + "=" * 70)
    print("LOADING MODEL AND TESTING PREDICTION...")
    pipe_loaded = ScompLinkPipeline("Loaded Model")
    pipe_loaded.load_model('./staging/example_11')
    
    # Test on new images
    test_images = np.array([X_images[i] for i in range(5)])
    predictions = pipe_loaded.predict(test_images)
    print(f"Test cluster assignments: {predictions}")
    print("✅ Model saved, loaded, and tested successfully!")
    print("=" * 70)
    
except ImportError as e:
    print("\n⚠️  Image/CV dependencies not installed!")
    print("Install with: pip install .[img]")
    print(f"Error: {e}")
