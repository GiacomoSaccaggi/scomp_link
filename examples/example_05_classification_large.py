#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example 5: Classification - Large Dataset (SGD/Gradient Boosting/Random Forest)
Use Case: >= 300 records per category with mixed features
"""

import pandas as pd
import numpy as np
from scomp_link import ScompLinkPipeline
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

np.random.seed(42)
n_per_class = 1000
categories = ['Low', 'Medium', 'High']
data = []

for cat in categories:
    for _ in range(n_per_class):
        if cat == 'Low':
            data.append([np.random.normal(30, 5), np.random.normal(30000, 5000), 
                        np.random.choice(['A', 'B', 'C']), cat])
        elif cat == 'Medium':
            data.append([np.random.normal(45, 5), np.random.normal(60000, 10000), 
                        np.random.choice(['B', 'C', 'D']), cat])
        else:
            data.append([np.random.normal(55, 5), np.random.normal(100000, 15000), 
                        np.random.choice(['C', 'D', 'E']), cat])

df = pd.DataFrame(data, columns=['age', 'income', 'segment', 'risk_level'])

print("=" * 70)
print("EXAMPLE 5: Classification - Large Dataset (SGD/Gradient Boosting)")
print("=" * 70)
print(f"Dataset size: {len(df)} records")

pipe = ScompLinkPipeline("Large Dataset Risk Classification")
pipe.set_objectives(["Maximize Accuracy", "Handle Large Dataset Efficiently"])
pipe.import_and_clean_data(df)
pipe.select_variables(target_col='risk_level')
pipe.choose_model("categorical_known", metadata={
    "records_per_category": 1000,
    "exogenous_type": "mixed"
})

models_to_test = {
    'SGD': {
        'model': SGDClassifier(random_state=42, max_iter=1000),
        'params_grid': {'loss': ['hinge', 'log_loss'], 'alpha': [0.0001, 0.001, 0.01]}
    },
    'GradientBoosting': {
        'model': GradientBoostingClassifier(random_state=42),
        'params_grid': {'n_estimators': [100, 200], 'max_depth': [3, 5], 'learning_rate': [0.01, 0.1]}
    },
    'RandomForest': {
        'model': RandomForestClassifier(random_state=42),
        'params_grid': {'n_estimators': [100, 200], 'max_depth': [10, 20]}
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
model_path = pipe.save_model('./staging/example_05')
print("=" * 70)

# Load model and predict
print("\n" + "=" * 70)
print("LOADING MODEL AND TESTING PREDICTION...")
pipe_loaded = ScompLinkPipeline("Loaded Model")
pipe_loaded.load_model('./staging/example_05')

test_data = df.sample(5)[['age', 'income', 'segment']]
predictions = pipe_loaded.predict(test_data)
print(f"Test predictions: {predictions[:5]}")
print("✅ Model saved, loaded, and tested successfully!")
print("=" * 70)
