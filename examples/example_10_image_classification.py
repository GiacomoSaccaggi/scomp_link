#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example 10: Image Classification - CNN
Use Case: Image classification with convolutional neural networks
"""

import pandas as pd
import numpy as np
from scomp_link import ScompLinkPipeline

print("=" * 70)
print("EXAMPLE 10: Image Classification - CNN")
print("=" * 70)

try:
    np.random.seed(42)
    n_samples = 300
    img_size = 28 * 28
    
    X_images = []
    y_labels = []
    
    for category in ['cat', 'dog', 'bird']:
        for _ in range(n_samples // 3):
            if category == 'cat':
                img = np.random.normal(0.3, 0.1, img_size)
            elif category == 'dog':
                img = np.random.normal(0.5, 0.1, img_size)
            else:
                img = np.random.normal(0.7, 0.1, img_size)
            
            X_images.append(img.reshape(28, 28, 1))
            y_labels.append(category)
    
    df = pd.DataFrame({'image_data': X_images, 'category': y_labels})
    
    print(f"Dataset size: {len(df)} records")
    print(f"Categories: {df['category'].unique().tolist()}")
    
    pipe = ScompLinkPipeline("Image Classification with CNN")
    pipe.set_objectives(["Maximize Accuracy", "Handle Image Data"])
    pipe.import_and_clean_data(df)
    pipe.select_variables(target_col='category')
    pipe.choose_model("categorical_known", metadata={"data_type": "images", "count_per_category": 100})
    
    results = pipe.run_pipeline(task_type="image", image_col='image_data', test_size=0.2)
    
    print("\n" + "=" * 70)
    print("RESULTS:")
    print(f"Model Type: {results['model_type']}")
    print(f"Accuracy: {results['metrics']['accuracy']:.3f}")
    print("=" * 70)
    
    print("\n✅ Image classification completed!")
    
    # Save model
    print("\n" + "=" * 70)
    print("SAVING MODEL...")
    model_path = pipe.save_model('./staging/example_10')
    print("=" * 70)
    
    # Load model and predict
    print("\n" + "=" * 70)
    print("LOADING MODEL AND TESTING PREDICTION...")
    pipe_loaded = ScompLinkPipeline("Loaded Model")
    pipe_loaded.load_model('./staging/example_10')
    
    test_images = np.array([df['image_data'].iloc[i] for i in range(3)])
    predictions = pipe_loaded.predict(test_images)
    print(f"Test predictions: {predictions}")
    print("✅ Model saved, loaded, and tested successfully!")
    print("=" * 70)
    
except ImportError as e:
    print("\n⚠️  Image/CV dependencies not installed!")
    print("Install with: pip install .[img]")
    print(f"Error: {e}")
