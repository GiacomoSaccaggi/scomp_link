# Complete Workflow Documentation

## Overview

This document maps the scomp-link implementation to the complete analysis workflow, showing how each component implements specific phases of the data science pipeline.

**Supported Python versions:** 3.10, 3.11, 3.12, 3.13

**Install:** `pip install scomp-link`

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

**Module:** `core.py` — `ScompLinkPipeline.__init__()`, `set_objectives()`

---

### Phase 2: Preprocessing (P1-P12)

#### P1-P2: Business & Data Understanding
```python
prep = Preprocessor(df)
summary = prep.run_eda()
```

#### P3: Data Acquisition
Manual data loading + `import_and_clean_data()`

#### P4: Data Cleaning
```python
pipe.import_and_clean_data(df)
# - Removes duplicates (with try/except for unhashable types)
# - Handles outliers
# - Manages missing values
```

#### P5: Data Integration (Record Linkage)
```python
prep = Preprocessor(df)
integrated_df = prep.integrate_data(external_df, on='key_column', how='inner')
```

#### P6: Data Selection
```python
pipe.select_variables(target_col='target', feature_cols=['feature1', 'feature2'])
```

#### P7: Data Transformation
Automatic transformation based on data types:
- Categorical → One-hot encoding
- Numerical → Standardization
- Text → TF-IDF or BERT tokenization
- Dates → Feature extraction

#### P10: Feature Selection
```python
prep = Preprocessor(df)
selected_features = prep.feature_selection(target_col='target', n_features=10)
```

#### P11: EDA - Knowledge Presentation
```python
summary = prep.run_eda()
```

#### P12: Dataset Preparation
```python
X_train, X_test, y_train, y_test = prep.prepare_datasets(target_col='target', test_size=0.2)
```

**Module:** `preprocessing/data_processor.py` — `Preprocessor` class

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
# Only numerical, all important → Ridge / SVR
pipe.choose_model("numerical_prediction", metadata={
    "only_numerical_exogenous": True,
    "all_variables_important": True
})

# Only numerical, feature selection needed → Lasso / Elastic Net
pipe.choose_model("numerical_prediction", metadata={
    "only_numerical_exogenous": True,
    "all_variables_important": False
})

# Mixed features → Gradient Boosting / Random Forest
pipe.choose_model("numerical_prediction", metadata={
    "only_numerical_exogenous": False
})
```

**> 100,000 records:**
```python
# Only numerical → SGD Regressor
# Mixed features → Gradient Boosting / Random Forest
```

#### Categorical Classification

**Images:**
```python
# < 500 per category → Pre-trained model
pipe.choose_model("categorical_known", metadata={
    "data_type": "images",
    "count_per_category": 300
})

# ≥ 500 per category → CNN (ResNet/Inception)
pipe.choose_model("categorical_known", metadata={
    "data_type": "images",
    "count_per_category": 600
})
```

**Text:**
```python
pipe.choose_model("categorical_known", metadata={"data_type": "text"})

# Contrastive Learning (BERT + FAISS)
results = pipe.run_pipeline(task_type="text", text_col='text', use_contrastive=True)

# TF-IDF + SGD (fast, simple)
results = pipe.run_pipeline(task_type="text", text_col='text', use_contrastive=False)
```

**Categorical Variables:**
```python
# < 5 features → Theoretical Psychometric Model
pipe.choose_model("categorical_known", metadata={
    "exogenous_type": "categorical",
    "num_features": 3
})

# ≥ 5 features → Naive Bayes / Classification Tree
pipe.choose_model("categorical_known", metadata={
    "exogenous_type": "categorical",
    "num_features": 10
})
```

**Mixed Variables:**
```python
# < 300 records per category → SVC / K-Neighbors / Naive Bayes
pipe.choose_model("categorical_known", metadata={"records_per_category": 200})

# ≥ 300 records per category → SGD / Gradient Boosting / Random Forest
pipe.choose_model("categorical_known", metadata={"records_per_category": 500})
```

#### Clustering

```python
# Categories known → KMeans / Hierarchical Clustering
pipe.choose_model("categorical_unknown", metadata={"categories_known": True})

# Categories unknown → Mean-Shift Clustering
pipe.choose_model("categorical_unknown", metadata={"categories_known": False})
```

#### Numerical Study

```python
# Geospatial → Geostatistical Model / Kriging
pipe.choose_model("numerical_study", metadata={"geospatial": True})

# Time series → UCM State Space
pipe.choose_model("numerical_study", metadata={"time_series": True})

# Other → Randomized PCA / Statistical Tests
pipe.choose_model("numerical_study")
```

#### Multi-Numerical Prediction

```python
# Time series → VAR / VARMA
pipe.choose_model("multi_numerical_prediction", metadata={"time_series": True})

# Other → MLP
pipe.choose_model("multi_numerical_prediction")
```

**Module:** `models/model_factory.py` — `ModelFactory.get_model()`

---

### Phase 4: Modeling (M1-M4)

#### M1: Missing Values Handling
```python
prep.clean_data(handle_missing='mean')  # or 'median', 'mode', 'drop'
```

#### M2: Outlier Management
```python
prep.clean_data(remove_outliers=True, outlier_threshold=3.0)
```

#### M3: Algorithm Parameters
```python
from scomp_link.models.regressor_optimizer import RegressorOptimizer

optimizer = RegressorOptimizer(
    df=df, y_col='target', x_cols=features,
    x_complexity_col=features[0],
    models_to_test=models_dict
)
optimizer.test_models_regression()
```

#### M4: Validation Parameters

**C1: LOOCV**
```python
results = pipe.run_pipeline(
    task_type="regression",
    advanced_cv=True,
    cv_methods=['loocv']
)
```

**C2: K-Fold**
```python
validator = Validator(model)
cv_scores = validator.k_fold_cv(X, y, k=5)
```

**C3: Bootstrap**
```python
results = pipe.run_pipeline(
    task_type="regression",
    advanced_cv=True,
    cv_methods=['bootstrap'],
    bootstrap_iterations=1000
)
```

**C4: Neural Network Epochs**
```python
results = pipe.run_pipeline(
    task_type="text", text_col='text',
    epochs=10, batch_size=32
)
```

**Modules:**
- `models/regressor_optimizer.py` — `RegressorOptimizer`
- `models/classifier_optimizer.py` — `ClassifierOptimizer`
- `validation/advanced_cv.py` — `AdvancedCV`

---

### Phase 5: Validation

#### V3: Evaluation Metrics

**Regression:**
```python
metrics = validator.evaluate(y_test, y_pred, task_type="regression")
# MSE, RMSE, MAE, R²
```

**Classification:**
```python
metrics = validator.evaluate(y_test, y_pred, task_type="classification")
# Accuracy, Precision, Recall, F1
```

**Module:** `validation/model_validator.py` — `Validator`

---

### Phase 6: Decision Flow

```
VALIDATION → {
    FAIL → Return to Model Selection
    SUCCESS → Ensemble Learning
}
```

#### SUCCESS Path — Ensemble Learning
```python
results = pipe.run_pipeline(
    task_type="regression",
    models_to_test=models_dict,
    use_ensemble=True,
    ensemble_strategy='voting'  # or 'stacking'
)
```

**Module:** `models/ensemble_optimizer.py` — `EnsembleOptimizer`

---

## Specialized Workflows

### Text Classification

```python
# Contrastive Learning (best for many classes, semantic similarity)
pipe.run_pipeline(
    task_type="text", text_col='text',
    use_contrastive=True,
    text_model='bert-base-uncased',
    epochs=3, batch_size=32
)

# TF-IDF + SGD (fast, simple)
pipe.run_pipeline(
    task_type="text", text_col='text',
    use_contrastive=False
)
```

**Save/Load:**
```python
pipe.save_model('./my_model')
pipe_loaded = ScompLinkPipeline("Loaded")
pipe_loaded.load_model('./my_model')
predictions = pipe_loaded.predict(["new text"])
```

### Text Clustering

```python
pipe.choose_model("categorical_unknown", metadata={"categories_known": True})
results = pipe.run_pipeline(
    task_type="text_clustering",
    text_col='text',
    n_clusters=5,
    text_model='bert-base-uncased'
)
```

### Image Classification

```python
pipe.choose_model("categorical_known", metadata={"data_type": "images"})
results = pipe.run_pipeline(task_type="image", image_col='pixels')
```

### Image Clustering

```python
results = pipe.run_pipeline(task_type="image_clustering", image_col='pixels', n_clusters=4)
```

### Anomaly Detection (Tabular)

```python
from scomp_link.models.anomaly_detector import AnomalyDetector

detector = AnomalyDetector(
    contamination=0.05,
    methods=['iforest', 'lof', 'tabnet', 'transformer'],
    consensus_threshold=2
)
results = detector.fit_predict(df, features=['col1', 'col2'])
```

### Anomaly Detection (Time Series)

```python
from scomp_link.models.ts_anomaly_detector import TimeSeriesAnomalyDetector

detector = TimeSeriesAnomalyDetector(
    methods=['autoencoder', 'moving_avg', 'moving_median', 'arima'],
    time_steps=50, window_size=30, n_sigma=3.0
)
detector.fit(normal_data)
results = detector.detect(test_data)
```

### HTML/PDF Reporting

```python
from scomp_link.utils.report_html import ScompLinkHTMLReport

report = ScompLinkHTMLReport('My Report')
report.add_title('Analysis Results')
report.add_text('Description...')
report.add_dataframe(df, 'Data Summary')
report.add_graph_to_report(fig, 'Chart Title')
report.open_section('Details')
report.add_text('Section content')
report.close_section()
report.save_html('report.html')
report.save_pdf('report.pdf')  # requires playwright
```

---

## Complete Pipeline Example

```python
from scomp_link import ScompLinkPipeline
import pandas as pd

# 1. PROBLEM IDENTIFICATION
pipe = ScompLinkPipeline("Sales Forecasting")

# 2. OBJECTIVES FORMULATION
pipe.set_objectives(["Minimize RMSE", "Maximize R²"])

# 3. PREPROCESSING (P3-P12)
df = pd.read_csv("sales_data.csv")
pipe.import_and_clean_data(df)
pipe.select_variables(target_col='sales')

# 4. MODEL SELECTION
pipe.choose_model("numerical_prediction", metadata={
    "only_numerical_exogenous": True,
    "all_variables_important": False
})

# 5. MODELING & VALIDATION
results = pipe.run_pipeline(
    task_type="regression",
    test_size=0.2,
    use_ensemble=True,
    ensemble_strategy='voting',
    advanced_cv=True,
    cv_methods=['bootstrap'],
    bootstrap_iterations=1000
)

# 6. RESULTS
print(f"Status: {results['status']}")
print(f"Metrics: {results['metrics']}")
print(f"Ensemble: {results['ensemble_scores']}")
print(f"Report: {results['report_path']}")
```

---

## Module Mapping

| Workflow Phase | Module | Class/Function |
|---------------|--------|----------------|
| Problem/Objectives | `core.py` | `ScompLinkPipeline` |
| P1-P12 Preprocessing | `preprocessing/data_processor.py` | `Preprocessor` |
| Model Selection | `models/model_factory.py` | `ModelFactory` |
| Regression Optimization | `models/regressor_optimizer.py` | `RegressorOptimizer` |
| Classification Optimization | `models/classifier_optimizer.py` | `ClassifierOptimizer` |
| Text (Contrastive) | `models/contrastive_text.py` | `ContrastiveTextClassifier` |
| Text (TF-IDF) | `core.py` | TfidfVectorizer + SGDClassifier pipeline |
| Text Clustering | `models/unsupervised_text.py` | `TextClusterer` |
| Image Classification | `models/supervised_img.py` | `CNNImg` |
| Image Clustering | `models/unsupervised_img.py` | `ClusterImg` |
| Anomaly (Tabular) | `models/anomaly_detector.py` | `AnomalyDetector` |
| Anomaly (Time Series) | `models/ts_anomaly_detector.py` | `TimeSeriesAnomalyDetector` |
| Ensemble | `models/ensemble_optimizer.py` | `EnsembleOptimizer` |
| Advanced CV | `validation/advanced_cv.py` | `AdvancedCV` |
| Evaluation | `validation/model_validator.py` | `Validator` |
| HTML/PDF Reports | `utils/report_html.py` | `ScompLinkHTMLReport` |
| Plotly Utilities | `utils/plotly_utils.py` | `histogram`, `barchart`, `linechart`, `area_chart` |

---

## Examples Mapping

| Example | Workflow |
|---------|----------|
| 01 | Small regression (< 1k) → Econometric Model |
| 02 | Medium regression + Lasso feature selection |
| 03 | Mixed features → Gradient Boosting |
| 04 | Small classification → SVC/K-Neighbors |
| 05 | Large classification → SGD/GB/RF |
| 06 | Clustering (known K) → KMeans |
| 07 | Clustering (unknown K) → Mean-Shift |
| 08 | Very large regression (> 100k) → SGD |
| 09 | Text → Contrastive Learning (BERT + FAISS) |
| 10 | Image classification → CNN |
| 11 | Image clustering → Feature extraction + KMeans |
| 12 | Text configuration (Contrastive vs TF-IDF) |
| 13 | Text clustering → Sentence-transformers + KMeans |
| 14 | Ensemble (Voting/Stacking) + Advanced CV |
| 15 | Tabular anomaly detection (4 methods + consensus) |
| 16 | Time series anomaly detection (AE + statistical) |
| 17 | HTML/PDF report generation |
| 18 | ContrastiveTextClassifier direct API |

---

## Project Structure

```
scomp_link/
├── scomp_link/                  # Main package
│   ├── core.py                  # Pipeline orchestrator
│   ├── __init__.py              # Package exports
│   ├── preprocessing/           # P1-P12
│   │   └── data_processor.py
│   ├── models/                  # Model implementations
│   │   ├── model_factory.py
│   │   ├── regressor_optimizer.py
│   │   ├── classifier_optimizer.py
│   │   ├── ensemble_optimizer.py
│   │   ├── contrastive_text.py
│   │   ├── contrastive_net.py
│   │   ├── supervised_text.py
│   │   ├── supervised_img.py
│   │   ├── unsupervised_text.py
│   │   ├── unsupervised_img.py
│   │   ├── anomaly_detector.py
│   │   └── ts_anomaly_detector.py
│   ├── validation/              # V1-V3, M4
│   │   ├── model_validator.py
│   │   └── advanced_cv.py
│   └── utils/                   # Reporting
│       ├── report_html.py
│       └── plotly_utils.py
├── examples/                    # 18 examples
├── tests/                       # Test suite
├── requirements/                # Per-version dependency files
├── pyproject.toml               # Source of truth for deps & config
├── setup.py                     # Minimal shim for editable installs
├── tox.ini                      # Multi-version testing (3.10-3.13)
├── VERSION_REQUIREMENTS.md      # Dependency version reference
└── WORKFLOW.md                  # This file
```

---

## See Also

- [README.md](README.md) — Quick start and API reference
- [VERSION_REQUIREMENTS.md](VERSION_REQUIREMENTS.md) — Dependency versions
- [examples/README.md](examples/README.md) — All 18 examples documented
- [Ensemble Learning](scomp_link/models/README_ENSEMBLE.md) — Voting & Stacking

---

📦 [scomp-link on PyPI](https://pypi.org/project/scomp-link/)
