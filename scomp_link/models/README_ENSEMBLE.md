# Ensemble Learning & Advanced Cross-Validation

## Overview

This module adds two powerful features to scomp-link:

1. **Ensemble Learning**: Combine multiple models using voting or stacking strategies
2. **Advanced Cross-Validation**: LOOCV and Bootstrap validation methods

## Features

### 1. Ensemble Learning

Combine multiple trained models to improve prediction accuracy and robustness.

**Strategies:**
- **Voting**: Average predictions from multiple models
- **Stacking**: Use a meta-learner to combine base model predictions

**Usage:**

```python
from scomp_link import ScompLinkPipeline

# Define multiple models
models_to_test = {
    'LinearRegression': {
        'model': LinearRegression(),
        'params_grid': {'fit_intercept': [True, False]}
    },
    'Ridge': {
        'model': Ridge(),
        'params_grid': {'alpha': [0.1, 1.0, 10.0]}
    },
    'RandomForest': {
        'model': RandomForestRegressor(),
        'params_grid': {'n_estimators': [50, 100]}
    }
}

# Run pipeline with ensemble
pipe = ScompLinkPipeline("Ensemble Demo")
pipe.import_and_clean_data(df)
pipe.select_variables(target_col='y')

results = pipe.run_pipeline(
    task_type="regression",
    models_to_test=models_to_test,
    use_ensemble=True,              # Enable ensemble
    ensemble_strategy='voting'       # or 'stacking'
)
```

### 2. Advanced Cross-Validation

Go beyond standard K-Fold with LOOCV and Bootstrap validation.

**Methods:**
- **LOOCV (Leave-One-Out)**: Train on N-1 samples, test on 1 (repeated N times)
- **Bootstrap**: Resample with replacement, evaluate on out-of-bag samples

**Usage:**

```python
results = pipe.run_pipeline(
    task_type="regression",
    models_to_test=models_to_test,
    advanced_cv=True,                    # Enable advanced CV
    cv_methods=['loocv', 'bootstrap'],   # Choose methods
    bootstrap_iterations=1000            # Bootstrap samples
)

# Access results
if 'advanced_cv' in results:
    for method, cv_result in results['advanced_cv'].items():
        print(f"{cv_result['method']}: {cv_result['mean_score']:.4f}")
```

## API Reference

### EnsembleOptimizer

```python
from scomp_link.models.ensemble_optimizer import EnsembleOptimizer

# Create ensemble
base_models = [('lr', LinearRegression()), ('ridge', Ridge())]
ensemble = EnsembleOptimizer(
    base_models=base_models,
    task_type='regression',  # or 'classification'
    strategy='voting'        # or 'stacking'
)

# Fit and predict
ensemble.fit(X_train, y_train)
predictions = ensemble.predict(X_test)

# Evaluate
scores = ensemble.evaluate_ensemble(X, y, cv=5)
```

### AdvancedCV

```python
from scomp_link.validation.advanced_cv import AdvancedCV

# LOOCV
loocv_results = AdvancedCV.loocv(model, X, y)

# Bootstrap
bootstrap_results = AdvancedCV.bootstrap(
    model, X, y, 
    n_iterations=1000,
    test_size=0.3
)

# Run all methods
all_results = AdvancedCV.evaluate_all(
    model, X, y,
    include_loocv=True,
    include_bootstrap=True,
    bootstrap_iterations=1000
)
```

## Performance Considerations

### LOOCV
- **Pros**: Unbiased, uses all data
- **Cons**: Computationally expensive (N model fits)
- **Recommendation**: Use only for datasets < 1000 samples

### Bootstrap
- **Pros**: Provides confidence intervals, parallelizable
- **Cons**: May underestimate variance
- **Recommendation**: Use 1000+ iterations for stable estimates

### Ensemble
- **Voting**: Fast, simple, works well with diverse models
- **Stacking**: More powerful but requires more computation

## Examples

See `examples/example_14_ensemble_advanced_cv.py` for complete working examples.

## Testing

Run tests:
```bash
pytest tests/test_ensemble_advanced_cv.py -v
```

## Integration with Core Pipeline

Both features are fully integrated into `ScompLinkPipeline.run_pipeline()`:

```python
results = pipe.run_pipeline(
    task_type="regression",
    models_to_test=models_to_test,
    
    # Ensemble options
    use_ensemble=True,
    ensemble_strategy='voting',  # or 'stacking'
    
    # Advanced CV options
    advanced_cv=True,
    cv_methods=['bootstrap'],
    bootstrap_iterations=1000
)
```

Results include:
- `optimizer_results`: Individual model performances
- `ensemble_scores`: Ensemble cross-validation scores
- `advanced_cv`: LOOCV/Bootstrap validation results
