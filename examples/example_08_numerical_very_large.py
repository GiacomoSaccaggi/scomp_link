#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example 8: Numerical Prediction - Very Large Dataset (SGD Regressor)
Use Case: > 100k records with only numerical features
"""

import pandas as pd
import numpy as np
from scomp_link import ScompLinkPipeline
from sklearn.linear_model import SGDRegressor

np.random.seed(42)
n = 150000
print("Generating large dataset...")
df = pd.DataFrame({
    'feature_1': np.random.randn(n),
    'feature_2': np.random.randn(n),
    'feature_3': np.random.randn(n),
    'feature_4': np.random.randn(n),
    'feature_5': np.random.randn(n),
    'target': np.random.randn(n)
})
df['target'] = (1.5*df['feature_1'] + 2*df['feature_2'] - 
                0.5*df['feature_3'] + np.random.randn(n)*0.3)

print("=" * 70)
print("EXAMPLE 8: Numerical Prediction - Very Large Dataset (SGD Regressor)")
print("=" * 70)
print(f"Dataset size: {len(df)} records")

pipe = ScompLinkPipeline("Large Scale Prediction with SGD")
pipe.set_objectives(["Minimize RMSE", "Handle Large Dataset Efficiently"])
pipe.import_and_clean_data(df)
pipe.select_variables(target_col='target')
pipe.choose_model("numerical_prediction", metadata={
    "only_numerical_exogenous": True,
    "all_variables_important": True
})

models_to_test = {
    'SGDRegressor': {
        'model': SGDRegressor(random_state=42, max_iter=1000),
        'params_grid': {
            'loss': ['squared_error', 'huber'],
            'alpha': [0.0001, 0.001],
            'learning_rate': ['constant', 'adaptive']
        }
    }
}

print("\nTraining on large dataset...")
results = pipe.run_pipeline(task_type="regression", test_size=0.2, models_to_test=models_to_test)

print("\n" + "=" * 70)
print("RESULTS:")
for model_name, model_results in results['optimizer_results'].items():
    print(f"\n{model_name}: {model_results['Params']}")
print("=" * 70)

# Save model
print("\n" + "=" * 70)
print("SAVING MODEL...")
model_path = pipe.save_model('./staging/example_08')
print("=" * 70)

# Load model and predict
print("\n" + "=" * 70)
print("LOADING MODEL AND TESTING PREDICTION...")
pipe_loaded = ScompLinkPipeline("Loaded Model")
pipe_loaded.load_model('./staging/example_08')

test_data = df.sample(5)[['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5']]
predictions = pipe_loaded.predict(test_data)
print(f"Test predictions: {predictions[:5]}")
print("✅ Model saved, loaded, and tested successfully!")
print("=" * 70)
