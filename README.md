# scomp-link: The Astromech Arm for Your Python Projects

## May the code be with you

[![CI](https://github.com/GiacomoSaccaggi/scomp-link/actions/workflows/ci.yml/badge.svg)](https://github.com/GiacomoSaccaggi/scomp-link/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/GiacomoSaccaggi/scomp-link/branch/main/graph/badge.svg)](https://codecov.io/gh/GiacomoSaccaggi/scomp-link)
[![Python](https://img.shields.io/pypi/pyversions/scomp-link)](https://pypi.org/project/scomp-link/)
[![License](https://img.shields.io/badge/license-MIT-blue)]()
[![PyPI](https://img.shields.io/pypi/v/scomp-link)](https://pypi.org/project/scomp-link/)

---

## Overview

**scomp-link** is an end-to-end machine learning toolkit that automates the complete ML workflow — from data profiling and preprocessing to model selection, training, validation, explainability, monitoring, and deployment.

It includes a **full-featured CLI** for zero-code ML workflows and a Python API for programmatic use.

---

## Installation

```bash
pip install scomp-link
```

Requires Python 3.10+.

---

## Key Features

| Category | Features |
|----------|----------|
| **Pipeline** | Automated model selection, training, validation, HTML reports |
| **CLI** | 13 commands — `run`, `predict`, `explain`, `engineer`, `forecast`, `anomaly`, `drift`, `fairness`, `quality`, `report`, `compare`, `info`, `init` |
| **Preprocessing** | Data cleaning, feature engineering (interactions, log, dates, target encoding, binning), data quality profiling |
| **Models** | Regression, classification, clustering, time series forecasting, anomaly detection, text (BERT contrastive), images (CNN) |
| **Tuning** | Optuna (Bayesian), Halving Grid Search, Early Stopping CV |
| **Validation** | K-Fold, LOOCV, Bootstrap, ensemble (voting/stacking) |
| **Explainability** | SHAP values, LIME explanations |
| **Monitoring** | Data drift detection (PSI + KS test) |
| **Fairness** | Demographic parity, disparate impact (4/5 rule), equalized odds |
| **Persistence** | Custom `.scomp` format (model + preprocessor + config + metrics + sample data) |
| **Reporting** | Interactive HTML reports (Plotly), data quality reports |

---

## CLI Quick Start

```bash
# Scaffold a new project
scomp-link init my_project

# Profile your data
scomp-link quality --data data.csv --output report.html

# Feature engineering
scomp-link engineer --data data.csv --target y --interactions --log-transform --output features.csv

# Train a model
scomp-link run --data features.csv --target y --task regression --save-artifact model.scomp

# Predict
scomp-link predict --artifact model.scomp --data new_data.csv --output predictions.csv

# Explain
scomp-link explain --artifact model.scomp --data test.csv

# Detect drift
scomp-link drift --reference train.csv --current production.csv

# Forecast time series
scomp-link forecast --data series.csv --column value --horizon 30

# Anomaly detection
scomp-link anomaly --data data.csv --methods iforest,lof

# Fairness check
scomp-link fairness --data preds.csv --target y_true --predicted y_pred --sensitive gender

# Compare models
scomp-link compare --artifacts v1.scomp v2.scomp

# Generate EDA report
scomp-link report --data data.csv --output eda_report.html

# Generate model evaluation report
scomp-link report --artifact model.scomp --data test.csv --output model_report.html
```

---

## Python API Quick Start

```python
from scomp_link import ScompLinkPipeline, ScompArtifact, set_verbosity
import pandas as pd

# Control output
set_verbosity("info")  # "silent" | "warning" | "info" | "debug"

# Build pipeline
pipe = ScompLinkPipeline("My Project")
pipe.set_objectives(["Minimize RMSE"])
pipe.import_and_clean_data(df)
pipe.select_variables(target_col='target')
pipe.choose_model("numerical_prediction")
results = pipe.run_pipeline(task_type="regression")

# Save as artifact
artifact = ScompArtifact()
artifact.set_model(pipe.model)
artifact.set_config(task_type='regression', target_col='target')
artifact.set_metrics(results['metrics'])
artifact.save('model.scomp')

# Load and predict
loaded = ScompArtifact.load('model.scomp')
predictions = loaded.predict(new_data)
```

---

## Feature Engineering

```python
from scomp_link import FeatureEngineer

fe = FeatureEngineer(
    interactions=True,      # Polynomial interactions
    log_transform=True,     # Log1p for skewed features
    date_features=True,     # Extract year/month/dow/weekend
    target_encode=True,     # Encode high-cardinality categoricals
    auto_bin=True,          # Quantile binning
)
X_train_eng = fe.fit_transform(X_train, y_train)
X_test_eng = fe.transform(X_test)
```

---

## Advanced Hyperparameter Tuning

```python
from scomp_link.models.advanced_tuning import OptunaOptimizer

def param_space(trial):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 20),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
    }

optimizer = OptunaOptimizer(GradientBoostingRegressor, param_space, scoring='r2', n_trials=100)
best_model = optimizer.optimize(X_train, y_train)
```

---

## Explainability

```python
from scomp_link import ShapExplainer, LimeExplainer

# SHAP
shap_exp = ShapExplainer(model, X_train[:100])
shap_exp.explain(X_test)
importance = shap_exp.feature_importance()
fig = shap_exp.plot_importance()

# LIME
lime_exp = LimeExplainer(model, X_train, task='regression')
exp = lime_exp.explain_instance(X_test.iloc[0])
fig = lime_exp.plot_explanation(exp)
```

---

## Data Drift Detection

```python
from scomp_link import DriftDetector

detector = DriftDetector(X_train, psi_threshold=0.2)
report = detector.detect(X_production)
summary = detector.summary(report)
fig = detector.plot_drift_report(report)
```

---

## Fairness & Bias Metrics

```python
from scomp_link import FairnessMetrics

fm = FairnessMetrics(y_true, y_pred, sensitive_feature=df['gender'])
report = fm.compute_all()
print(fm.summary(report))
fig = fm.plot_fairness_report(report)
```

---

## Time Series Forecasting

```python
from scomp_link import TimeSeriesForecaster

fc = TimeSeriesForecaster(method='auto', horizon=30)
fc.fit(series)
forecast = fc.predict_with_ci()
cv_results = fc.walk_forward_cv(series, n_splits=5)
fig = fc.plot_forecast()
```

---

## Data Quality Report

```python
from scomp_link import DataQualityReport

dqr = DataQualityReport(df)
report = dqr.generate()  # missing, cardinality, constants, duplicates, correlations
dqr.save_html('quality_report.html')
```

---

## Anomaly Detection

```python
from scomp_link import AnomalyDetector

detector = AnomalyDetector(
    contamination=0.05,
    methods=['iforest', 'lof', 'tabnet', 'transformer'],
    consensus_threshold=2,
)
results = detector.fit_predict(df, features=['col1', 'col2', 'col3'])
```

---

## Project Structure

```
scomp_link/
├── cli.py                    # CLI (12 commands)
├── core.py                   # ScompLinkPipeline orchestrator
├── preprocessing/
│   ├── data_processor.py     # Preprocessor (polars backend)
│   ├── feature_engineer.py   # FeatureEngineer (sklearn-compatible)
│   └── data_quality.py       # DataQualityReport
├── models/
│   ├── model_factory.py      # Decision-tree model selection
│   ├── regressor_optimizer.py
│   ├── classifier_optimizer.py
│   ├── ensemble_optimizer.py
│   ├── advanced_tuning.py    # Optuna, Halving, EarlyStopping
│   ├── forecaster.py         # TimeSeriesForecaster
│   ├── anomaly_detector.py
│   ├── ts_anomaly_detector.py
│   ├── contrastive_text.py   # BERT contrastive learning
│   ├── supervised_text.py
│   └── supervised_img.py
├── validation/
│   ├── model_validator.py    # Metrics + HTML reports
│   ├── advanced_cv.py        # LOOCV, Bootstrap
│   └── fairness.py           # FairnessMetrics
├── explainability/
│   └── explainer.py          # ShapExplainer, LimeExplainer
├── monitoring/
│   └── drift_detector.py     # DriftDetector (PSI + KS)
├── persistence/
│   └── artifact.py           # ScompArtifact (.scomp format)
└── utils/
    ├── logger.py             # Configurable logging
    ├── report_html.py        # HTML report builder
    └── plotly_utils.py       # Chart utilities
```

---

## Testing

```bash
# Run all tests (185 tests)
pytest tests/ -v

# With coverage
pytest tests/ --cov=scomp_link --cov-report=html
```

---

## Documentation

Full documentation with API reference and CLI guide:

```bash
pip install mkdocs mkdocs-material "mkdocstrings[python]"
mkdocs serve  # http://localhost:8000
```

---

## Contributing

```bash
git clone https://github.com/GiacomoSaccaggi/scomp-link.git
cd scomp_link
pip install -e ".[dev]"
pytest tests/ -v
```

---

## License

MIT License

---

**May the code be with you.** 🚀

📦 [scomp-link on PyPI](https://pypi.org/project/scomp-link/)
