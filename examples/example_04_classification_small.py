#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example 4: Classification - Small Dataset per Category (SVC/K-Neighbors)
Use Case: < 300 records per category with mixed features
"""

import pandas as pd
import numpy as np
from scomp_link import ScompLinkPipeline
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

np.random.seed(42)
n_per_class = 200
categories = ['A', 'B', 'C']
data = []

for cat in categories:
    for _ in range(n_per_class):
        if cat == 'A':
            data.append([np.random.normal(0, 1), np.random.normal(0, 1), cat])
        elif cat == 'B':
            data.append([np.random.normal(3, 1), np.random.normal(3, 1), cat])
        else:
            data.append([np.random.normal(-3, 1), np.random.normal(3, 1), cat])

df = pd.DataFrame(data, columns=['feature_1', 'feature_2', 'category'])

print("=" * 70)
print("EXAMPLE 4: Classification - Small Dataset (SVC/K-Neighbors)")
print("=" * 70)
print(f"Dataset size: {len(df)} records")

pipe = ScompLinkPipeline("Small Dataset Classification")
pipe.set_objectives(["Maximize Accuracy", "Maximize F1 Score"])
pipe.import_and_clean_data(df)
pipe.select_variables(target_col='category')
pipe.choose_model("categorical_known", metadata={
    "records_per_category": 200,
    "exogenous_type": "mixed"
})

models_to_test = {
    'SVC': {
        'model': SVC(probability=True, random_state=42),
        'params_grid': {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}
    },
    'KNeighbors': {
        'model': KNeighborsClassifier(),
        'params_grid': {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
    },
    'NaiveBayes': {
        'model': GaussianNB(),
        'params_grid': {}
    }
}

results = pipe.run_pipeline(task_type="classification", test_size=0.2, models_to_test=models_to_test)

print("\n" + "=" * 70)
print("RESULTS:")
for model_name, model_results in results['optimizer_results'].items():
    print(f"\n{model_name}: {model_results['Params']}")
print("=" * 70)

# Save model
print("\n" + "=" * 70)
print("SAVING MODEL...")
model_path = pipe.save_model('./staging/example_04')
print("=" * 70)

# Load model and predict
print("\n" + "=" * 70)
print("LOADING MODEL AND TESTING PREDICTION...")
pipe_loaded = ScompLinkPipeline("Loaded Model")
pipe_loaded.load_model('./staging/example_04')

test_data = df.sample(5)[['feature_1', 'feature_2']]
predictions = pipe_loaded.predict(test_data)
print(f"Test predictions: {predictions[:5]}")
print("✅ Model saved, loaded, and tested successfully!")
print("=" * 70)
