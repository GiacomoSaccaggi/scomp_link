# scomp-link

**The Astromech Arm for Your Python Projects** 🚀

[![CI](https://github.com/GiacomoSaccaggi/scomp-link/actions/workflows/ci.yml/badge.svg)](https://github.com/GiacomoSaccaggi/scomp-link/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/GiacomoSaccaggi/scomp-link/branch/main/graph/badge.svg)](https://codecov.io/gh/GiacomoSaccaggi/scomp-link)
[![Python](https://img.shields.io/pypi/pyversions/scomp-link)](https://pypi.org/project/scomp-link/)
[![PyPI](https://img.shields.io/pypi/v/scomp-link)](https://pypi.org/project/scomp-link/)

---

## What is scomp-link?

**scomp-link** is an end-to-end machine learning toolkit that automates the complete ML workflow — from data profiling and preprocessing to model selection, training, validation, explainability, monitoring, and deployment.

It provides:

- A **CLI** with 12 commands for zero-code ML workflows
- A **Python API** for programmatic use and integration
- A custom **`.scomp` format** for saving and loading complete pipelines

---

## Key Features

| Category | What you get |
|----------|-------------|
| 🎯 **Pipeline** | Automated model selection, training, validation, HTML reports |
| ⌨️ **CLI** | 13 commands: `run`, `predict`, `explain`, `engineer`, `forecast`, `anomaly`, `drift`, `fairness`, `quality`, `report`, `compare`, `info`, `init` |
| 🔧 **Preprocessing** | Polars-backed cleaning, feature engineering (interactions, log, dates, target encoding, binning), data quality profiling |
| 🤖 **Models** | Regression, classification, clustering, time series forecasting, anomaly detection, NLP (BERT contrastive), images (CNN) |
| ⚡ **Tuning** | Optuna (Bayesian), Halving Grid Search, Early Stopping CV |
| ✅ **Validation** | K-Fold, LOOCV, Bootstrap, ensemble (voting/stacking) |
| 🔬 **Explainability** | SHAP values, LIME explanations |
| 📊 **Monitoring** | Data drift detection (PSI + KS test) |
| ⚖️ **Fairness** | Demographic parity, disparate impact (4/5 rule), equalized odds |
| 💾 **Persistence** | Custom `.scomp` format (model + preprocessor + config + metrics + sample data) |
| 📋 **Reporting** | Interactive HTML reports (Plotly), data quality reports |

---

## Installation

```bash
pip install scomp-link
```

Requires Python 3.10+.

---

## Quick Start (CLI)

```bash
# Scaffold a project
scomp-link init my_project

# Profile data
scomp-link quality --data data.csv --output report.html

# Train a model
scomp-link run --data data.csv --target y --task regression --save-artifact model.scomp

# Predict
scomp-link predict --artifact model.scomp --data new_data.csv --output predictions.csv
```

See the full [CLI Reference](cli.md) for all 12 commands.

---

## Quick Start (Python)

```python
from scomp_link import ScompLinkPipeline, ScompArtifact, set_verbosity

set_verbosity("info")  # "silent" | "warning" | "info" | "debug"

pipe = ScompLinkPipeline("My Project")
pipe.set_objectives(["Minimize RMSE"])
pipe.import_and_clean_data(df)
pipe.select_variables(target_col='target')
pipe.choose_model("numerical_prediction")
results = pipe.run_pipeline(task_type="regression")

# Save complete pipeline
artifact = ScompArtifact()
artifact.set_model(pipe.model)
artifact.set_config(task_type='regression', target_col='target')
artifact.set_metrics(results['metrics'])
artifact.save('model.scomp')
```

---

## Typical Workflow

```
┌─────────┐    ┌──────────┐    ┌─────────┐    ┌──────────┐    ┌─────────┐
│ Quality │───▶│ Engineer │───▶│  Train  │───▶│ Explain  │───▶│ Deploy  │
│ Report  │    │ Features │    │  Model  │    │  Model   │    │ Monitor │
└─────────┘    └──────────┘    └─────────┘    └──────────┘    └─────────┘
                                                                    │
                                                               ┌────▼────┐
                                                               │  Drift  │
                                                               │  Check  │
                                                               └─────────┘
```

Each step is available as both a CLI command and a Python class.

---

## Next Steps

- [Quick Start Guide](quickstart.md) — step-by-step tutorial
- [CLI Reference](cli.md) — all 12 commands with options and examples
- [API Reference](api/pipeline.md) — Python class documentation
- [Examples](examples.md) — 26 standalone runnable examples
- [Changelog](changelog.md) — version history
