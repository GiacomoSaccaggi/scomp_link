#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example 1: Numerical Prediction - Small Dataset (<1000 records)
Use Case: Econometric modeling for limited data scenarios
"""

import pandas as pd
import numpy as np
from scomp_link import ScompLinkPipeline

np.random.seed(42)
n = 500
df = pd.DataFrame({
    'income': np.random.normal(50000, 15000, n),
    'age': np.random.randint(22, 65, n),
    'education_years': np.random.randint(12, 20, n),
    'experience': np.random.randint(0, 30, n),
    'salary': np.random.normal(60000, 20000, n)
})
df['salary'] = df['income'] * 0.3 + df['age'] * 500 + df['education_years'] * 2000 + np.random.normal(0, 5000, n)

print("=" * 70)
print("EXAMPLE 1: Numerical Prediction - Small Dataset (Econometric Model)")
print("=" * 70)
print(f"Dataset size: {len(df)} records")

pipe = ScompLinkPipeline("Salary Prediction with Limited Data")
pipe.set_objectives(["Minimize RMSE", "Maximize R²"])
pipe.import_and_clean_data(df)
pipe.select_variables(target_col='salary')
pipe.choose_model("numerical_prediction", metadata={
    "only_numerical_exogenous": True,
    "all_variables_important": True
})

results = pipe.run_pipeline(task_type="regression", test_size=0.2)

print("\n" + "=" * 70)
print("RESULTS:")
print(f"Model Type: {results['model_type']}")
print(f"Metrics: {results['metrics']}")
print(f"Report: {results['report_path']}")
print("=" * 70)

# Save model
print("\n" + "=" * 70)
print("SAVING MODEL...")
model_path = pipe.save_model('./staging/example_01')
print("=" * 70)

# Load model and predict
print("\n" + "=" * 70)
print("LOADING MODEL AND TESTING PREDICTION...")
pipe_loaded = ScompLinkPipeline("Loaded Model")
pipe_loaded.load_model('./staging/example_01')

# Test prediction on new data
test_data = df.sample(5)[['income', 'age', 'education_years', 'experience']]
predictions = pipe_loaded.predict(test_data)
print(f"Test predictions: {predictions[:5]}")
print("✅ Model saved, loaded, and tested successfully!")
print("=" * 70)
