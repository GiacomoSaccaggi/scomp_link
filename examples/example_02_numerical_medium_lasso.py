#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example 2: Numerical Prediction - Medium Dataset with Feature Selection
Use Case: Lasso/Elastic Net for datasets with 1k-100k records
"""

import pandas as pd
import numpy as np
from scomp_link import ScompLinkPipeline
from sklearn.linear_model import Lasso, ElasticNet

# Generate medium dataset (1k-100k records)
np.random.seed(42)
n = 5000
df = pd.DataFrame({
    'feature_1': np.random.randn(n),
    'feature_2': np.random.randn(n),
    'feature_3': np.random.randn(n),
    'feature_4': np.random.randn(n),
    'feature_5': np.random.randn(n),
    'noise_1': np.random.randn(n),
    'noise_2': np.random.randn(n),
    'target': np.random.randn(n)
})
df['target'] = 2*df['feature_1'] + 3*df['feature_2'] - df['feature_3'] + np.random.randn(n)*0.5

print("=" * 70)
print("EXAMPLE 2: Numerical Prediction - Medium Dataset (Lasso/Elastic Net)")
print("=" * 70)
print(f"Dataset size: {len(df)} records")

pipe = ScompLinkPipeline("Feature Selection for Regression")
pipe.set_objectives(["Minimize RMSE", "Select Important Features"])
pipe.import_and_clean_data(df)
pipe.select_variables(target_col='target')
pipe.choose_model("numerical_prediction", metadata={
    "only_numerical_exogenous": True,
    "all_variables_important": False
})

models_to_test = {
    'Lasso': {
        'model': Lasso(),
        'params_grid': {'alpha': [0.01, 0.1, 1.0, 10.0]}
    },
    'ElasticNet': {
        'model': ElasticNet(),
        'params_grid': {'alpha': [0.1, 1.0], 'l1_ratio': [0.3, 0.5, 0.7]}
    }
}

results = pipe.run_pipeline(task_type="regression", test_size=0.2, models_to_test=models_to_test)

print("\n" + "=" * 70)
print("RESULTS:")
for model_name, model_results in results['optimizer_results'].items():
    print(f"\n{model_name}:")
    print(f"  Best Params: {model_results['Params']}")
    print(f"  RMSE: {model_results.get('RMSE', 'N/A')}")
print("=" * 70)

# Save model
print("\n" + "=" * 70)
print("SAVING MODEL...")
model_path = pipe.save_model('./staging/example_02')
print("=" * 70)

# Load model and predict
print("\n" + "=" * 70)
print("LOADING MODEL AND TESTING PREDICTION...")
pipe_loaded = ScompLinkPipeline("Loaded Model")
pipe_loaded.load_model('./staging/example_02')

test_data = df.sample(5)[['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5', 'noise_1', 'noise_2']]
predictions = pipe_loaded.predict(test_data)
print(f"Test predictions: {predictions[:5]}")
print("✅ Model saved, loaded, and tested successfully!")
print("=" * 70)
