# -*- coding: utf-8 -*-
"""
Example: Ensemble Learning and Advanced Cross-Validation

This example demonstrates:
1. Training multiple models with RegressorOptimizer
2. Creating ensemble models (voting/stacking)
3. Advanced cross-validation (LOOCV, Bootstrap)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from scomp_link import ScompLinkPipeline

# Generate synthetic data
np.random.seed(42)
N = 500
X1 = np.random.randn(N)
X2 = np.random.randn(N)
X3 = np.random.randn(N)
y = 2*X1 + 3*X2 - 1.5*X3 + np.random.randn(N) * 0.5

df = pd.DataFrame({
    'x1': X1,
    'x2': X2,
    'x3': X3,
    'y': y
})

print("=" * 60)
print("EXAMPLE: Ensemble Learning + Advanced Cross-Validation")
print("=" * 60)

# Initialize pipeline
pipe = ScompLinkPipeline("Ensemble and Advanced CV Demo")
pipe.set_objectives(["Minimize RMSE", "Test Ensemble", "Test Advanced CV"])
pipe.import_and_clean_data(df)
pipe.select_variables(target_col='y')

# Define multiple models to test
models_to_test = {
    'LinearRegression': {
        'model': LinearRegression(),
        'params_grid': {
            'fit_intercept': [True, False]
        }
    },
    'Ridge': {
        'model': Ridge(),
        'params_grid': {
            'alpha': [0.1, 1.0, 10.0]
        }
    },
    'Lasso': {
        'model': Lasso(),
        'params_grid': {
            'alpha': [0.1, 1.0, 10.0]
        }
    },
    'RandomForest': {
        'model': RandomForestRegressor(random_state=42),
        'params_grid': {
            'n_estimators': [50, 100],
            'max_depth': [5, 10]
        }
    }
}

print("\n" + "=" * 60)
print("STEP 1: Training Multiple Models")
print("=" * 60)

# Run pipeline WITH ensemble and advanced CV
results = pipe.run_pipeline(
    task_type="regression",
    models_to_test=models_to_test,
    test_size=0.2,
    use_ensemble=True,              # Enable ensemble learning
    ensemble_strategy='voting',      # Use voting ensemble
    advanced_cv=True,                # Enable advanced CV
    cv_methods=['bootstrap'],        # Use bootstrap (LOOCV too slow for demo)
    bootstrap_iterations=500         # 500 bootstrap samples
)

print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

# Display optimizer results
if 'optimizer_results' in results:
    print("\nIndividual Model Results:")
    for model_name, model_result in results['optimizer_results'].items():
        print(f"  - {model_name}: {model_result['Params']}")

# Display ensemble results
if 'ensemble_scores' in results:
    print("\nEnsemble Performance:")
    print(f"  Mean CV Score: {results['ensemble_scores']['mean_score']:.4f}")
    print(f"  Std CV Score: {results['ensemble_scores']['std_score']:.4f}")

# Display advanced CV results
if 'advanced_cv' in results and results['advanced_cv']:
    print("\nAdvanced Cross-Validation Results:")
    for method, cv_result in results['advanced_cv'].items():
        print(f"\n  {cv_result['method']}:")
        print(f"    Mean Score: {cv_result['mean_score']:.4f}")
        print(f"    Std Score: {cv_result['std_score']:.4f}")
        if 'confidence_interval_95' in cv_result:
            ci = cv_result['confidence_interval_95']
            print(f"    95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")

print("\n" + "=" * 60)
print("EXAMPLE 2: Stacking Ensemble")
print("=" * 60)

# Test with stacking strategy
pipe2 = ScompLinkPipeline("Stacking Ensemble Demo")
pipe2.import_and_clean_data(df)
pipe2.select_variables(target_col='y')

results2 = pipe2.run_pipeline(
    task_type="regression",
    models_to_test=models_to_test,
    test_size=0.2,
    use_ensemble=True,
    ensemble_strategy='stacking',    # Use stacking ensemble
    advanced_cv=False                # Skip advanced CV for speed
)

if 'ensemble_scores' in results2:
    print("\nStacking Ensemble Performance:")
    print(f"  Mean CV Score: {results2['ensemble_scores']['mean_score']:.4f}")
    print(f"  Std CV Score: {results2['ensemble_scores']['std_score']:.4f}")

print("\n✅ Example completed successfully!")
print("📊 Check 'ScompLink_Validation_Report.html' for detailed results")
