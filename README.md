# scomp-link: The Astromech Arm for Your Python Projects

## May the code be with you

[![CI](https://github.com/GiacomoSaccaggi/scomp_link/actions/workflows/ci.yml/badge.svg)](https://github.com/GiacomoSaccaggi/scomp_link/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/GiacomoSaccaggi/scomp_link/branch/main/graph/badge.svg)](https://codecov.io/gh/GiacomoSaccaggi/scomp_link)
[![PyPI](https://img.shields.io/pypi/v/scomp-link)](https://pypi.org/project/scomp-link/)
[![Python](https://img.shields.io/pypi/pyversions/scomp-link)](https://pypi.org/project/scomp-link/)
[![Downloads](https://img.shields.io/pypi/dm/scomp-link)](https://pypi.org/project/scomp-link/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/GiacomoSaccaggi/scomp_link/blob/main/LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Typed: pyright](https://img.shields.io/badge/typing-pyright-blue)](https://github.com/microsoft/pyright)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit)](https://pre-commit.com)
[![Security: CodeQL](https://img.shields.io/badge/security-CodeQL-green?logo=github)](https://github.com/GiacomoSaccaggi/scomp_link/security)
[![Docker](https://img.shields.io/docker/v/jack15121/scomp-link?label=docker&logo=docker)](https://hub.docker.com/r/jack15121/scomp-link)
[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue?logo=github)](https://giacomosaccaggi.github.io/scomp_link/)
[![MCP](https://img.shields.io/badge/MCP-22_tools-blue?logo=anthropic)](https://modelcontextprotocol.io)
[![Smithery](https://smithery.ai/badge/giacomosaccaggi/scomp-link)](https://smithery.ai/servers/giacomosaccaggi/scomp-link)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97-MCP%20Space-yellow)](https://huggingface.co/spaces/Euribor512/scomp-link)
[![Claude](https://img.shields.io/badge/Works_with-Claude-blueviolet?logo=anthropic)](AGENT_INTEGRATION.md)
[![Kiro](https://img.shields.io/badge/Works_with-Kiro-orange)](AGENT_INTEGRATION.md)
[![Cursor](https://img.shields.io/badge/Works_with-Cursor-green)](AGENT_INTEGRATION.md)


<!-- mcp-name: io.github.giacomosaccaggi/scomp-link -->
---

## Overview

**scomp-link** is an end-to-end machine learning toolkit that automates the complete ML workflow — from data profiling and preprocessing to model selection, training, validation, explainability, monitoring, and deployment.

It includes a **full-featured CLI** for zero-code ML workflows and a Python API for programmatic use.

---

## Installation

```bash
pip install scomp-link
```

Requires Python 3.10+. Import is near-instant (~6ms) thanks to lazy loading — heavy dependencies load only when needed. Python 3.14 is supported experimentally (TensorFlow not yet available on 3.14).

---

## Key Features

| Category | Features |
|----------|----------|
| **Pipeline** | Automated model selection, training, validation, HTML reports |
| **CLI** | 26 commands — `run`, `predict`, `text`, `embed`, `cluster`, `tune`, `validate`, `explain`, `engineer`, `forecast`, `anomaly`, `drift`, `fairness`, `quality`, `describe`, `report`, `compare`, `monitor`, `serve`, `export`, `pipeline`, `info`, `init`, `init-config`, `list-models`, `check-deps` |
| **Preprocessing** | Data cleaning, feature engineering (Polars backend — interactions, log, dates, target encoding, binning), data quality profiling |
| **Models** | Regression, classification, clustering, time series forecasting, anomaly detection, text (BERT contrastive + weak learner head), images (CNN) |
| **Tuning** | Optuna (Bayesian), Halving Grid Search, Early Stopping CV |
| **Validation** | K-Fold, LOOCV, Bootstrap, ensemble (voting/stacking) |
| **Explainability** | SHAP values, LIME explanations |
| **Monitoring** | Data drift detection (PSI + KS test) |
| **Fairness** | Demographic parity, disparate impact (4/5 rule), equalized odds |
| **Persistence** | Custom `.scomp` format (model + preprocessor + config + metrics + sample data) |
| **Visualization** | 31 RAWGraphs SVG charts, Plotly interactive, Highcharts, centralized color system |
| **Reporting** | Interactive HTML reports with embedded charts, data quality reports |

---

## CLI Quick Start

```bash
# Scaffold a new project
scomp-link init my_project

# Quick dataset profiling
scomp-link describe --data data.csv --format table

# Full data quality report
scomp-link quality --data data.csv --output report.html

# Feature engineering
scomp-link engineer --data data.csv --target y --interactions --log-transform --output features.csv

# Train a model
scomp-link run --data features.csv --target y --task regression --save-artifact model.scomp

# Train with text data
scomp-link text --data tickets.csv --text-col message --target category --method contrastive --head auto

# Extract embeddings from trained model
scomp-link embed --data new_texts.csv --text-col message --artifact model.scomp --output embeddings.npy

# Clustering
scomp-link cluster --data customers.csv --n-clusters 5 --plot clusters.html

# Hyperparameter tuning
scomp-link tune --data train.csv --target y --task regression --method optuna --n-trials 100

# Predict
scomp-link predict --artifact model.scomp --data new_data.csv --output predictions.csv

# Validate on test data
scomp-link validate --artifact model.scomp --data test.csv --target y --report report.html

# Explain
scomp-link explain --artifact model.scomp --data test.csv

# Detect drift
scomp-link drift --reference train.csv --current production.csv --plot drift.html

# Production monitoring (drift + quality + performance)
scomp-link monitor --reference train.csv --current prod.csv --artifact model.scomp --target y

# Forecast time series
scomp-link forecast --data series.csv --column value --horizon 30 --plot forecast.html

# Anomaly detection
scomp-link anomaly --data data.csv --methods iforest,lof,tabnet,transformer

# Fairness check
scomp-link fairness --data preds.csv --target y_true --predicted y_pred --sensitive gender

# Compare models
scomp-link compare --artifacts v1.scomp v2.scomp --plot comparison.html

# Run full pipeline from YAML config
scomp-link pipeline --config pipeline.yaml

# Serve model as REST API
scomp-link serve --artifact model.scomp --port 8080

# Export model to standard format
scomp-link export --artifact model.scomp --format onnx

# Generate reports
scomp-link report --data data.csv --output eda_report.html
scomp-link report --artifact model.scomp --data test.csv --output model_report.html

# Utilities
scomp-link list-models
scomp-link check-deps

# Configuration
scomp-link init-config              # Create global config (~/.scomp-link/config.yaml)
scomp-link init-config --local      # Create project-level config (.scomp-link.yaml)
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
├── cli.py                    # CLI (24 commands)
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
    ├── colors.py             # Centralized color palettes
    ├── logger.py             # Configurable logging
    ├── report_html.py        # HTML report builder
    ├── plotly_utils.py       # Plotly chart utilities
    ├── highcharts.py         # Highcharts visualizations
    └── rawgraphs/            # 31 SVG chart functions (server-side)
```

---


---

## AI Agent Integration

scomp-link works natively with AI agents via **MCP (Model Context Protocol)** and **Agent Skills**.

### MCP Server (22 tools for structured agent calls)

```bash
pip install scomp-link[mcp]
scomp-link mcp  # Starts MCP server (stdio transport)
```

**Claude Desktop** (`claude_desktop_config.json`):
```json
{"mcpServers": {"scomp-link": {"command": "scomp-link", "args": ["mcp"]}}}
```

**Cursor** (plugin — one click):

[![Add to Cursor](https://img.shields.io/badge/Add_to-Cursor-green?logo=cursor)](https://cursor.com/marketplace/scomp-link)

**Kiro** (`.kiro/mcp.json`):
```json
{"mcpServers": {"scomp-link": {"command": "scomp-link", "args": ["mcp"]}}}
```

**Remote (no install needed)** — connect to the hosted MCP server on [🤗 Hugging Face Space](https://huggingface.co/spaces/Euribor512/scomp-link) or [🔧 Smithery](https://smithery.ai/servers/giacomosaccaggi/scomp-link):
```json
{"mcpServers": {"scomp-link": {"url": "https://Euribor512-scomp-link.hf.space/sse"}}}
```

**Docker:**
```bash
docker pull jack15121/scomp-link:latest
docker run -i jack15121/scomp-link mcp
```

**Available tools:** `describe_data`, `train_model`, `predict`, `validate_model`, `detect_drift`, `detect_anomalies`, `check_fairness`, `forecast_series`, `engineer_features`, `cluster_data`, `generate_report`, `create_visualization`, `compare_models`, `export_model`, `embed_text`, `select_backbone`, `report_create`, `report_add_section`, `report_add_text`, `report_add_table`, `report_add_chart`, `report_save`

### Quick Setup Prompt for AI Agents

Copy-paste this prompt into your AI agent (Claude, ChatGPT, Cursor, etc.) to enable scomp-link capabilities:

> You have access to scomp-link, an ML toolkit with 22 MCP tools. Use them for:
> - **Data profiling**: `describe_data(path)` — always start here
> - **Training**: `train_model(data, target, task)` with optional `tune=true` for Optuna
> - **Validation**: `validate_model(artifact, data, target)` for test evaluation
> - **Reports**: Use the report builder for custom dashboards:
>   1. `report_create(title)` → get report_id
>   2. `report_add_section(id, title)` → structure
>   3. `report_add_chart(id, engine, type, data, title)` → 39 chart types (plotly/rawgraphs/highcharts)
>   4. `report_add_table(id, json_data, title)` → data tables
>   5. `report_save(id, path)` → save HTML
> - **Monitoring**: `detect_drift`, `detect_anomalies`, `check_fairness`
> - **Forecasting**: `forecast_series(data, column, horizon)`
>
> Report branding defaults come from `~/.scomp-link/config.yaml`. Run `scomp-link init-config` to set up.

### Agent Skill (zero-dependency documentation)

```bash
# For Kiro
cp -r skills/scomp-link ~/.kiro/skills/

# For Claude Code
cp -r skills/scomp-link .claude/skills/
```

See [AGENT_INTEGRATION.md](AGENT_INTEGRATION.md) for full setup guide.

---

## Report Builder (MCP)

Build fully custom branded HTML reports step-by-step via MCP tools:

```python
# 1. Configure corporate defaults (one-time setup)
# scomp-link init-config
# Edit ~/.scomp-link/config.yaml with your branding

# 2. Create a report (uses config defaults automatically)
report_create("Q4 Performance Report")

# 3. Add content
report_add_section(report_id, "Executive Summary")
report_add_table(report_id, metrics_json, "Key Metrics")
report_add_chart(report_id, "plotly", "linechart", data, "Revenue Trend")
report_add_chart(report_id, "rawgraphs", "treemap", data, "Market Share")

# 4. Save
report_save(report_id, "q4_report.html")
```

**Configuration precedence:** `.scomp-link.yaml` (local) > `~/.scomp-link/config.yaml` (global) > built-in defaults

```bash
# Create global config with your corporate branding
scomp-link init-config

# Or create project-level config
scomp-link init-config --local
```

---

## Testing

```bash
# Run all tests
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
