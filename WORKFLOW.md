# Complete Workflow Documentation

## Overview

This document maps the scomp-link implementation to the complete analysis workflow, showing how each component implements specific phases of the data science pipeline.

## Workflow Phases

### Phase 1: Problem Identification & Objectives

**Workflow Steps:**
- Problem Identification
- Objectives Formulation
- Analysis Development

**Implementation:**
```python
from scomp_link import ScompLinkPipeline

# Problem Identification
pipe = ScompLinkPipeline("Customer Churn Prediction")

# Objectives Formulation
pipe.set_objectives([
    "Maximize Accuracy",
    "Minimize False Negatives",
    "Interpretable Model"
])
```

**Module:** `core.py` - `ScompLinkPipeline.__init__()`, `set_objectives()`

---

### Phase 2: Preprocessing (P1-P12)

#### P1-P2: Business & Data Understanding
```python
# Automatic data profiling
prep = Preprocessor(df)
summary = prep.run_eda()
```

#### P3: Data Acquisition
- Internal data
- Open data
- Web scraping
- Data acquisition

**Implementation:** Manual data loading + `import_and_clean_data()`

#### P4: Data Cleaning
```python
# Automatic cleaning
pipe.import_and_clean_data(df)
# - Removes duplicates
# - Handles outliers
# - Manages missing values
```

#### P5: Data Integration (Record Linkage)
```python
prep = Preprocessor(df)
integrated_df = prep.integrate_data(
    external_df,
    on='key_column',
    how='inner'
)
```

#### P6: Data Selection
```python
# Select relevant features
pipe.select_variables(
    target_col='target',
    feature_cols=['feature1', 'feature2']
)
```

#### P7: Data Transformation
Automatic transformation based on data types:
- Categorical → One-hot encoding
- Numerical → Standardization
- Text → Tokenization
- Dates → Feature extraction

#### P8: Data Mining
Implemented through model training phase

#### P9: Relationship Evaluation
Implemented through validation metrics

#### P10: Feature Selection
```python
# Boruta algorithm
prep = Preprocessor(df)
selected_features = prep.feature_selection(
    target_col='target',
    n_features=10
)
```

#### P11: EDA - Knowledge Presentation
```python
summary = prep.run_eda()
# Generates statistics and visualizations
```

#### P12: Dataset Preparation
```python
X_train, X_test, y_train, y_test = prep.prepare_datasets(
    target_col='target',
    test_size=0.2
)
```

**Module:** `preprocessing/data_processor.py` - `Preprocessor` class

---

### Phase 3: Model Selection (Decision Tree)

#### Numerical Prediction

**< 1,000 records:**
```python
pipe.choose_model("numerical_prediction")
# → Econometric Model
```

**1,000 - 100,000 records:**
```python
# Only numerical, all important
pipe.choose_model("numerical_prediction", metadata={
    "only_numerical_exogenous": True,
    "all_variables_important": True
})
# → Ridge / SVR

# Only numerical, feature selection needed
pipe.choose_model("numerical_prediction", metadata={
    "only_numerical_exogenous": True,
    "all_variables_important": False
})
# → Lasso / Elastic Net

# Mixed features
pipe.choose_model("numerical_prediction", metadata={
    "only_numerical_exogenous": False
})
# → Gradient Boosting / Random Forest
```

**> 100,000 records:**
```python
# Only numerical
# → SGD Regressor

# Mixed features
# → Gradient Boosting / Random Forest
```

#### Categorical Classification

**Images:**
```python
pipe.choose_model("categorical_known", metadata={
    "data_type": "images",
    "count_per_category": 300  # < 500
})
# → Pre-trained model

pipe.choose_model("categorical_known", metadata={
    "data_type": "images",
    "count_per_category": 600  # ≥ 500
})
# → CNN (ResNet/Inception)
```

**Categorical Variables:**
```python
# Few variables
pipe.choose_model("categorical_known", metadata={
    "exogenous_type": "categorical",
    "num_features": 3  # < 5
})
# → Theoretical Psychometric Model

# Many variables
pipe.choose_model("categorical_known", metadata={
    "exogenous_type": "categorical",
    "num_features": 10  # ≥ 5
})
# → Naive Bayes / Classification Tree
```

**Mixed Variables:**
```python
# < 300 records per category
pipe.choose_model("categorical_known", metadata={
    "records_per_category": 200
})
# → SVC / K-Neighbors / Naive Bayes

# ≥ 300 records per category
pipe.choose_model("categorical_known", metadata={
    "records_per_category": 500
})
# → SGD / Gradient Boosting / Random Forest
```

#### Clustering

```python
# Categories unknown
pipe.choose_model("categorical_unknown", metadata={
    "categories_known": False
})
# → Mean-Shift Clustering

# Categories known
pipe.choose_model("categorical_unknown", metadata={
    "categories_known": True
})
# → KMeans / Hierarchical Clustering
```

#### Numerical Study

```python
# Geospatial data
pipe.choose_model("numerical_study", metadata={
    "geospatial": True
})
# → Geostatistical Model / Kriging

# Time series
pipe.choose_model("numerical_study", metadata={
    "time_series": True
})
# → UCM State Space

# Other
pipe.choose_model("numerical_study")
# → Randomized PCA / Statistical Tests
```

#### Multi-Numerical Prediction

```python
# Time series
pipe.choose_model("multi_numerical_prediction", metadata={
    "time_series": True
})
# → VAR / VARMA

# Other
pipe.choose_model("multi_numerical_prediction")
# → Multilayer Perceptron (MLP)
```

**Module:** `models/model_factory.py` - `ModelFactory.get_model()`

---

### Phase 4: Modeling (M1-M4)

#### M1: Missing Values Handling
```python
# Automatic in preprocessing
prep.clean_data(handle_missing='mean')  # or 'median', 'mode', 'drop'
```

#### M2: Outlier Management
```python
prep.clean_data(
    remove_outliers=True,
    outlier_threshold=3.0  # Z-score threshold
)
```

#### M3: Algorithm Parameters
```python
# Grid search optimization
from scomp_link.models import RegressorOptimizer

optimizer = RegressorOptimizer(
    df=df,
    y_col='target',
    x_cols=features,
    models_to_test=models_dict
)
optimizer.test_models_regression()
```

#### M4: Validation Parameters

**C1: Leave-One-Out Cross Validation (LOOCV)**
```python
from scomp_link.validation import AdvancedCV

results = AdvancedCV.loocv(model, X, y)
```

**C2: K-Fold Cross Validation**
```python
validator = Validator(model)
cv_scores = validator.k_fold_cv(X, y, k=5)
```

**C3: Bootstrap**
```python
results = AdvancedCV.bootstrap(
    model, X, y,
    n_iterations=1000
)
```

**C4: Neural Network Epochs**
```python
# For deep learning models
classifier.train_contrastive(
    df,
    epochs=10,
    validation_split=0.2
)
```

**Modules:**
- `models/regressor_optimizer.py` - `RegressorOptimizer`
- `models/classifier_optimizer.py` - `ClassifierOptimizer`
- `validation/advanced_cv.py` - `AdvancedCV`

---

### Phase 5: Validation

#### V1: Interpretation vs Flexibility

**High Interpretability:**
- Linear Regression
- Logistic Regression
- Decision Trees

**High Flexibility:**
- Random Forest
- Gradient Boosting
- Neural Networks

#### V2: Underfitting vs Overfitting

```python
# Automatic detection
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)

if train_score > 0.95 and test_score < 0.7:
    # Overfitting → Return to model selection
    pass
elif train_score < 0.6:
    # Underfitting → Return to model selection
    pass
```

#### V3: Evaluation Metrics

**Regression:**
```python
metrics = validator.evaluate(y_test, y_pred, task_type="regression")
# - MSE, RMSE, MAE, R², MAPE
```

**Classification:**
```python
metrics = validator.evaluate(y_test, y_pred, task_type="classification")
# - Accuracy, Precision, Recall, F1, ROC AUC
```

**Module:** `validation/model_validator.py` - `Validator`

---

### Phase 6: Decision Flow

```
VALIDATION → {
    FAIL → Return to Model Selection (choose_model)
    SUCCESS → Ensemble Learning → Reinforcement Learning
}
```

#### FAIL Path
```python
# Automatic retry with different model
if metrics['r2'] < 0.5:
    pipe.choose_model("numerical_prediction", metadata={
        "only_numerical_exogenous": False  # Try mixed approach
    })
    results = pipe.run_pipeline(task_type="regression")
```

#### SUCCESS Path - Ensemble Learning
```python
results = pipe.run_pipeline(
    task_type="regression",
    models_to_test=models_dict,
    use_ensemble=True,
    ensemble_strategy='voting'  # or 'stacking'
)
```

**Module:** `models/ensemble_optimizer.py` - `EnsembleOptimizer`

---

## Complete Pipeline Example

```python
from scomp_link import ScompLinkPipeline
import pandas as pd

# 1. PROBLEM IDENTIFICATION
pipe = ScompLinkPipeline("Sales Forecasting")

# 2. OBJECTIVES FORMULATION
pipe.set_objectives(["Minimize RMSE", "Maximize R²"])

# 3. PREPROCESSING (P1-P12)
df = pd.read_csv("sales_data.csv")
pipe.import_and_clean_data(df)  # P3-P4
pipe.select_variables(target_col='sales')  # P6

# 4. MODEL SELECTION
pipe.choose_model("numerical_prediction", metadata={
    "only_numerical_exogenous": True,
    "all_variables_important": False
})

# 5. MODELING & VALIDATION (M1-M4, V1-V3)
results = pipe.run_pipeline(
    task_type="regression",
    test_size=0.2,
    use_ensemble=True,
    advanced_cv=True,
    cv_methods=['bootstrap'],
    bootstrap_iterations=1000
)

# 6. RESULTS
print(f"Status: {results['status']}")
print(f"Model: {results['model_type']}")
print(f"Metrics: {results['metrics']}")
print(f"Ensemble Score: {results['ensemble_scores']}")
print(f"Advanced CV: {results['advanced_cv']}")
print(f"Report: {results['report_path']}")
```

---

## Module Mapping

| Workflow Phase | Module | Class/Function |
|---------------|--------|----------------|
| Problem/Objectives | `core.py` | `ScompLinkPipeline.__init__()`, `set_objectives()` |
| P1-P12 Preprocessing | `preprocessing/` | `Preprocessor` |
| Model Selection | `models/model_factory.py` | `ModelFactory` |
| M3 Optimization | `models/*_optimizer.py` | `RegressorOptimizer`, `ClassifierOptimizer` |
| M4 Validation | `validation/advanced_cv.py` | `AdvancedCV` |
| V1-V3 Evaluation | `validation/model_validator.py` | `Validator` |
| Ensemble | `models/ensemble_optimizer.py` | `EnsembleOptimizer` |
| Reporting | `utils/report_html.py` | `ScompLinkHTMLReport` |

---

## Documentation Structure

```
scomp_link/
├── README.md                          # Main documentation
├── WORKFLOW.md                        # This file
├── scomp_link/
│   ├── preprocessing/README.md        # P1-P12 documentation
│   ├── models/README.md               # Model selection documentation
│   ├── models/README_ENSEMBLE.md      # Ensemble & Advanced CV
│   ├── validation/README.md           # Validation documentation
│   └── utils/README.md                # Utilities documentation
└── examples/
    └── example_14_ensemble_advanced_cv.py  # Complete example
```

---

## See Also

- [Main README](README.md) - Quick start and API reference
- [Preprocessing](scomp_link/preprocessing/README.md) - P1-P12 phases
- [Models](scomp_link/models/README.md) - Decision tree implementation
- [Validation](scomp_link/validation/README.md) - V1-V3 and M4
- [Ensemble Learning](scomp_link/models/README_ENSEMBLE.md) - Advanced features
- [Utilities](scomp_link/utils/README.md) - Reporting and visualization
