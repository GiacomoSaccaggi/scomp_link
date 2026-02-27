#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example: Using ContrastiveTextClassifier for Text Classification

This example demonstrates how to use the generalized contrastive learning
approach for text classification tasks.
"""

import pandas as pd
import numpy as np
from scomp_link.models.contrastive_text import ContrastiveTextClassifier

# ============= Example 1: Simple Text Classification =============
print("=" * 60)
print("Example 1: Simple Text Classification")
print("=" * 60)

# Create synthetic dataset
texts = [
    "New smartphone released with advanced AI features",
    "Football team wins championship after dramatic final",
    "Government announces new economic policy",
    "New movie breaks box office records",
    "Scientists discover new species in deep ocean"
]

labels = ['Technology', 'Sports', 'Politics', 'Entertainment', 'Science']

df = pd.DataFrame({'text': texts, 'category': labels})
df = pd.concat([df] * 20, ignore_index=True)

print(f"\nDataset: {len(df)} samples, {df['category'].nunique()} categories")

# Initialize and train
classifier = ContrastiveTextClassifier(model_name='bert-base-uncased', embedding_dim=128)
classifier.train_contrastive(df, text_col='text', label_col='category', epochs=3, batch_size=16)

# Test predictions
test_texts = ["Artificial intelligence transforms software development"]
result = classifier.predict(test_texts[0], top_k=3, return_confidence=True)
print(f"\nPrediction: {result['predictions'][0]} (confidence: {result['confidences'][0]:.3f})")

# Save model
classifier.save('./models/text_classifier')

print("\n✅ Example completed!")
