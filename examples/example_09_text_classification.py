#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example 9: Text Classification - Contrastive Learning
Use Case: Multi-class text classification with semantic understanding
"""

import pandas as pd
import numpy as np
from scomp_link import ScompLinkPipeline

print("=" * 70)
print("EXAMPLE 9: Text Classification - Contrastive Learning")
print("=" * 70)

try:
    texts = [
        "New AI model breaks performance records",
        "Stock market reaches all-time high",
        "Scientists discover new exoplanet",
        "Football team wins championship",
        "New smartphone features advanced camera",
        "Economic growth exceeds expectations",
        "Breakthrough in quantum computing",
        "Basketball player sets new record",
        "Latest laptop offers better performance",
        "Central bank adjusts interest rates"
    ] * 50
    
    labels = [
        'Technology', 'Business', 'Science', 'Sports', 'Technology',
        'Business', 'Science', 'Sports', 'Technology', 'Business'
    ] * 50
    
    df = pd.DataFrame({'text': texts, 'category': labels})
    
    print(f"Dataset size: {len(df)} records")
    print(f"Categories: {df['category'].unique().tolist()}")
    
    pipe = ScompLinkPipeline("Text Classification with Contrastive Learning")
    pipe.set_objectives(["Maximize Accuracy", "Semantic Understanding"])
    pipe.import_and_clean_data(df)
    pipe.select_variables(target_col='category')
    pipe.choose_model("categorical_known", metadata={"data_type": "text"})
    
    results = pipe.run_pipeline(task_type="text", text_col='text', epochs=3, batch_size=32, test_size=0.2)
    
    print("\n" + "=" * 70)
    print("RESULTS:")
    print(f"Model Type: {results['model_type']}")
    print(f"Status: {results['status']}")
    print(f"Metrics: {results['metrics']}")
    print("=" * 70)
    
    print("\n✅ Text classification completed!")
    
    # Save model
    print("\n" + "=" * 70)
    print("SAVING MODEL...")
    model_path = pipe.save_model('./staging/example_09')
    print("=" * 70)
    
    # Load model and predict
    print("\n" + "=" * 70)
    print("LOADING MODEL AND TESTING PREDICTION...")
    pipe_loaded = ScompLinkPipeline("Loaded Model")
    pipe_loaded.load_model('./staging/example_09')
    
    test_texts = ["AI breakthrough in healthcare", "Market shows positive trends"]
    predictions = pipe_loaded.predict(test_texts)
    print(f"Test predictions: {predictions}")
    print("✅ Model saved, loaded, and tested successfully!")
    print("=" * 70)
    
except ImportError as e:
    print("\n⚠️  NLP dependencies not installed!")
    print("Install with: pip install .[nlp]")
    print(f"Error: {e}")
