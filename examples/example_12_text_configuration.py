#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example 12: Text Classification - Configuration Options
Use Case: Shows how to configure text classification models
"""

import pandas as pd
import numpy as np
from scomp_link import ScompLinkPipeline

print("=" * 70)
print("EXAMPLE 12: Text Classification - Configuration Options")
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
    ] * 30
    
    labels = [
        'Technology', 'Business', 'Science', 'Sports', 'Technology',
        'Business', 'Science', 'Sports', 'Technology', 'Business'
    ] * 30
    
    df = pd.DataFrame({'text': texts, 'category': labels})
    
    print(f"Dataset size: {len(df)} records")
    print(f"Categories: {df['category'].unique().tolist()}")
    
    # OPTION 1: Contrastive Learning (default, best for many classes)
    print("\n" + "=" * 70)
    print("OPTION 1: Contrastive Learning")
    print("=" * 70)
    
    pipe1 = ScompLinkPipeline("Contrastive Text Classification")
    pipe1.set_objectives(["Maximize Accuracy", "Semantic Understanding"])
    pipe1.import_and_clean_data(df)
    pipe1.select_variables(target_col='category')
    pipe1.choose_model("categorical_known", metadata={"data_type": "text"})
    
    results1 = pipe1.run_pipeline(
        task_type="text",
        text_col='text',
        epochs=2,
        batch_size=32,
        test_size=0.2,
        text_model='bert-base-uncased',  # Can use: distilbert-base-uncased, roberta-base, etc.
        use_contrastive=True
    )
    
    print(f"Model Type: {results1['model_type']}")
    print(f"Status: {results1['status']}")
    
    # OPTION 2: Supervised Text Classifier (simpler, faster)
    print("\n" + "=" * 70)
    print("OPTION 2: Supervised Text Classifier")
    print("=" * 70)
    
    pipe2 = ScompLinkPipeline("Supervised Text Classification")
    pipe2.set_objectives(["Maximize Accuracy", "Fast Training"])
    pipe2.import_and_clean_data(df)
    pipe2.select_variables(target_col='category')
    pipe2.choose_model("categorical_known", metadata={"data_type": "text"})
    
    results2 = pipe2.run_pipeline(
        task_type="text",
        text_col='text',
        test_size=0.2,
        text_model='distilbert-base-uncased',  # Faster model
        text_max_length=64,  # Shorter sequences
        use_contrastive=False  # Use simple supervised approach
    )
    
    print(f"Model Type: {results2['model_type']}")
    print(f"Accuracy: {results2['metrics']['accuracy']:.3f}")
    
    print("\n" + "=" * 70)
    print("CONFIGURATION OPTIONS:")
    print("=" * 70)
    print("text_model options:")
    print("  - 'bert-base-uncased' (default, good accuracy)")
    print("  - 'distilbert-base-uncased' (faster, smaller)")
    print("  - 'roberta-base' (better performance)")
    print("  - 'albert-base-v2' (memory efficient)")
    print("\nuse_contrastive:")
    print("  - True: Best for many classes, semantic similarity")
    print("  - False: Faster training, simpler approach")
    print("\ntext_max_length:")
    print("  - 64: Short texts (tweets, titles)")
    print("  - 128: Medium texts (paragraphs)")
    print("  - 512: Long texts (documents)")
    print("=" * 70)
    
    print("\n✅ Text classification completed!")
    
    # Save model
    print("\n" + "=" * 70)
    print("SAVING MODEL...")
    model_path = pipe2.save_model('./staging/example_12')
    print("=" * 70)
    
    # Load model and predict
    print("\n" + "=" * 70)
    print("LOADING MODEL AND TESTING PREDICTION...")
    pipe_loaded = ScompLinkPipeline("Loaded Model")
    pipe_loaded.load_model('./staging/example_12')
    
    test_texts = ["New AI model released", "Stock prices increase"]
    predictions = pipe_loaded.predict(test_texts)
    print(f"Test predictions: {predictions}")
    print("✅ Model saved, loaded, and tested successfully!")
    print("=" * 70)
    
except ImportError as e:
    print("\n⚠️  NLP dependencies not installed!")
    print("Install with: pip install .[nlp]")
    print(f"Error: {e}")
