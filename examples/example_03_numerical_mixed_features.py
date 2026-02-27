#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Example 3: Numerical Prediction - Mixed Features (Gradient Boosting/Random Forest)
Use Case: Medium/large datasets with categorical and numerical features
"""

import pandas as pd
import numpy as np
from scomp_link import ScompLinkPipeline
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

np.random.seed(42)
n = 10000
df = pd.DataFrame({
    'age': np.random.randint(18, 80, n),
    'income': np.random.normal(50000, 20000, n),
    'credit_score': np.random.randint(300, 850, n),
    'city': np.random.choice(['NYC', 'LA', 'Chicago', 'Houston'], n),
    'employment_type': np.random.choice(['Full-time', 'Part-time', 'Self-employed'], n),
    'loan_amount': np.random.normal(200000, 50000, n)
})
df['loan_amount'] = (df['income'] * 3 + df['credit_score'] * 100 + 
                     df['age'] * 1000 + np.random.normal(0, 10000, n))

print("=" * 70)
print("EXAMPLE 3: Numerical Prediction - Mixed Features (Gradient Boosting)")
print("=" * 70)
print(f"Dataset size: {len(df)} records")

pipe = ScompLinkPipeline("Loan Amount Prediction with Mixed Features")
pipe.set_objectives(["Minimize RMSE", "Handle Mixed Data Types"])
pipe.import_and_clean_data(df)
pipe.select_variables(target_col='loan_amount')
pipe.choose_model("numerical_prediction", metadata={
    "only_numerical_exogenous": False,
    "all_variables_important": True
})

models_to_test = {
    'GradientBoosting': {
        'model': GradientBoostingRegressor(random_state=42),
        'params_grid': {
            'n_estimators': [100, 200],
            'max_depth': [3, 5],
            'learning_rate': [0.01, 0.1]
        }
    },
    'RandomForest': {
        'model': RandomForestRegressor(random_state=42),
        'params_grid': {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None]
        }
    }
}

results = pipe.run_pipeline(task_type="regression", test_size=0.2, models_to_test=models_to_test)

print("\n" + "=" * 70)
print("RESULTS:")
for model_name, model_results in results['optimizer_results'].items():
    print(f"\n{model_name}:")
    print(f"  Best Params: {model_results['Params']}")
print("=" * 70)

# Save model
print("\n" + "=" * 70)
print("SAVING MODEL...")
model_path = pipe.save_model('./staging/example_03')
print("=" * 70)

# Load model and predict
print("\n" + "=" * 70)
print("LOADING MODEL AND TESTING PREDICTION...")
pipe_loaded = ScompLinkPipeline("Loaded Model")
pipe_loaded.load_model('./staging/example_03')

test_data = df.sample(5)[['age', 'income', 'credit_score', 'city', 'employment_type']]
predictions = pipe_loaded.predict(test_data)
print(f"Test predictions: {predictions[:5]}")
print("✅ Model saved, loaded, and tested successfully!")
print("=" * 70)
