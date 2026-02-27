# Validation Module

## Overview

The validation module implements the complete evaluation workflow (VALUTAZIONE) from the analysis pipeline, including metrics calculation, cross-validation strategies, and automated report generation.

## Validation Workflow

This module implements the validation phase from the complete workflow:

```
TEST → VALUTAZIONE
├── V1: Interpretation vs Flexibility
├── V2: Underfitting vs Overfitting
└── V3: Model Evaluation Metrics

VALUTAZIONE → {
    FAIL → Return to Model Selection
    SUCCESS → Ensemble Learning → Reinforcement Learning
}
```

## Core Components

### Validator

Main class for model evaluation and validation:

```python
from scomp_link.validation import Validator
from sklearn.linear_model import LinearRegression

# Train model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Create validator
validator = Validator(model)

# Evaluate (V3: Metrics)
metrics = validator.evaluate(
    y_test, y_pred,
    task_type="regression"
)

print(f"RMSE: {metrics['rmse']:.4f}")
print(f"R²: {metrics['r2']:.4f}")
print(f"MAE: {metrics['mae']:.4f}")

# K-Fold Cross Validation
cv_scores = validator.k_fold_cv(X, y, k=5)
print(f"CV Mean: {cv_scores.mean():.4f}")
print(f"CV Std: {cv_scores.std():.4f}")

# Generate HTML report
validator.generate_validation_report(
    y_test, y_pred,
    task_type="regression",
    report_name="Validation_Report.html"
)
```

### AdvancedCV

Advanced cross-validation strategies (M4: Validation Parameters):

```python
from scomp_link.validation import AdvancedCV

# Leave-One-Out Cross Validation (C1)
loocv_results = AdvancedCV.loocv(model, X, y)
print(f"LOOCV Score: {loocv_results['mean_score']:.4f}")

# Bootstrap Validation (C3)
bootstrap_results = AdvancedCV.bootstrap(
    model, X, y,
    n_iterations=1000,
    test_size=0.3
)
print(f"Bootstrap Score: {bootstrap_results['mean_score']:.4f}")
print(f"95% CI: {bootstrap_results['confidence_interval_95']}")

# Run all validation methods
all_results = AdvancedCV.evaluate_all(
    model, X, y,
    include_loocv=True,
    include_bootstrap=True,
    bootstrap_iterations=1000
)
```

## Validation Strategies (M4)

### C1: Leave-One-Out Cross Validation (LOOCV)

Most rigorous validation for small datasets:

```python
# Automatic LOOCV (skips if dataset > 1000 samples)
results = AdvancedCV.loocv(model, X, y)

# Results include:
# - mean_score: Average performance
# - std_score: Standard deviation
# - n_splits: Number of iterations (= dataset size)
# - scores: Individual fold scores
```

**When to use:**
- Small datasets (< 1000 samples)
- Maximum use of training data
- Unbiased performance estimate

**Limitations:**
- Computationally expensive (N model fits)
- High variance in estimates

### C2: K-Fold Cross Validation

Standard validation strategy:

```python
# K-Fold CV (default in optimizers)
cv_scores = validator.k_fold_cv(X, y, k=5)

# Stratified K-Fold for classification
from sklearn.model_selection import StratifiedKFold
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True)
```

**When to use:**
- Medium to large datasets
- Balance between bias and variance
- Standard practice (k=5 or k=10)

### C3: Bootstrap Validation

Resampling-based validation with confidence intervals:

```python
bootstrap_results = AdvancedCV.bootstrap(
    model, X, y,
    n_iterations=1000,  # Number of bootstrap samples
    test_size=0.3,      # Out-of-bag test size
    random_state=42
)

# Access confidence intervals
ci_lower, ci_upper = bootstrap_results['confidence_interval_95']
print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")
```

**When to use:**
- Need confidence intervals
- Assess model stability
- Non-parametric validation

**Advantages:**
- Provides uncertainty estimates
- Works with any metric
- Robust to outliers

### C4: Neural Network Validation

For deep learning models (epochs and iterations):

```python
# Handled automatically in specialized models
from scomp_link.models.contrastive_text import ContrastiveTextClassifier

classifier = ContrastiveTextClassifier()
classifier.train_contrastive(
    df,
    text_col='text',
    label_col='category',
    epochs=10,              # C4: Epoch selection
    validation_split=0.2    # Validation set
)
```

## Evaluation Metrics (V3)

### Regression Metrics

```python
metrics = validator.evaluate(y_test, y_pred, task_type="regression")

# Available metrics:
# - mse: Mean Squared Error
# - rmse: Root Mean Squared Error
# - mae: Mean Absolute Error
# - r2: R² Score (coefficient of determination)
# - mape: Mean Absolute Percentage Error
```

### Classification Metrics

```python
metrics = validator.evaluate(
    y_test, y_pred,
    task_type="classification",
    y_proba=y_proba  # Probability predictions
)

# Available metrics:
# - accuracy: Overall accuracy
# - precision: Precision score
# - recall: Recall score
# - f1: F1 score
# - roc_auc: ROC AUC score (if y_proba provided)
# - confusion_matrix: Confusion matrix
```

### Clustering Metrics

```python
from sklearn.metrics import silhouette_score, davies_bouldin_score

silhouette = silhouette_score(X, clusters)
davies_bouldin = davies_bouldin_score(X, clusters)
```

## Model Evaluation Criteria

### V1: Interpretation vs Flexibility

Trade-off between model interpretability and flexibility:

**High Interpretability:**
- Linear Regression
- Logistic Regression
- Decision Trees
- Naive Bayes

**High Flexibility:**
- Random Forest
- Gradient Boosting
- Neural Networks
- SVM with RBF kernel

### V2: Underfitting vs Overfitting

Automatic detection through validation:

```python
# Training vs Test performance
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

if train_score > 0.95 and test_score < 0.7:
    print("⚠️  Overfitting detected")
elif train_score < 0.6 and test_score < 0.6:
    print("⚠️  Underfitting detected")
else:
    print("✅ Good fit")
```

**Overfitting indicators:**
- High training score, low test score
- Large gap between CV folds
- High model complexity

**Underfitting indicators:**
- Low training and test scores
- Simple model for complex data
- Insufficient features

## Automated Reporting

### HTML Validation Reports

Comprehensive reports with interactive visualizations:

```python
validator.generate_validation_report(
    y_test, y_pred,
    task_type="regression",
    y_proba=None,
    report_name="Validation_Report.html"
)
```

**Report includes:**

**For Regression:**
- Metrics summary (MSE, RMSE, MAE, R²)
- Observed vs Predicted scatter plot
- Residuals distribution histogram
- Residuals analysis with confidence intervals
- Binned residuals plot

**For Classification:**
- Metrics summary (Accuracy, F1, Precision, Recall)
- Confusion matrix heatmap
- Confidence distribution per class
- ROC curves (if probabilities provided)
- Classification report

**Interactive features:**
- Plotly-based visualizations
- Zoom, pan, hover tooltips
- Export to CSV
- Responsive design

## Integration with Pipeline

```python
from scomp_link import ScompLinkPipeline

pipe = ScompLinkPipeline("Project")
pipe.import_and_clean_data(df)
pipe.select_variables(target_col='y')
pipe.choose_model("numerical_prediction")

# Automatic validation with advanced CV
results = pipe.run_pipeline(
    task_type="regression",
    test_size=0.2,
    advanced_cv=True,
    cv_methods=['loocv', 'bootstrap'],
    bootstrap_iterations=1000
)

# Access validation results
print(results['metrics'])
print(results['advanced_cv'])
print(f"Report: {results['report_path']}")
```

## Decision Flow

```
Model Training → Validation → {
    FAIL (poor metrics) → Return to Model Selection
    SUCCESS (good metrics) → Consider Ensemble Learning
}
```

The pipeline automatically:
1. Evaluates model on test set
2. Runs advanced cross-validation (if enabled)
3. Generates HTML report
4. Returns metrics and validation results
5. Suggests ensemble if multiple models available

## Best Practices

1. **Always use cross-validation**: Don't rely on single train/test split
2. **Choose appropriate CV**: LOOCV for small data, K-Fold for large
3. **Check for overfitting**: Compare train vs test performance
4. **Use bootstrap for CI**: Get confidence intervals on metrics
5. **Generate reports**: Visual inspection is crucial
6. **Consider ensemble**: If single model fails, try ensemble

## Performance Considerations

### LOOCV
- **Time**: O(N × model_fit_time)
- **Use when**: N < 1000
- **Automatically skipped**: For large datasets

### Bootstrap
- **Time**: O(iterations × model_fit_time)
- **Recommended**: 1000+ iterations
- **Parallelizable**: Can be optimized

### K-Fold
- **Time**: O(k × model_fit_time)
- **Standard**: k=5 or k=10
- **Best balance**: Speed vs accuracy

## Dependencies

- scikit-learn: Metrics and CV strategies
- numpy: Numerical operations
- pandas: Data handling
- plotly: Interactive visualizations
- matplotlib: Static plots

## See Also

- [Models](../models/README.md)
- [Preprocessing](../preprocessing/README.md)
- [Ensemble Learning](../models/README_ENSEMBLE.md)
- [Complete Pipeline](../README.md)
