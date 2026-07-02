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

- A **CLI** with 25 commands for zero-code ML workflows
- A **Python API** for programmatic use and integration
- A custom **`.scomp` format** for saving and loading complete pipelines
- **39 chart types** across 3 visualization engines (Plotly, RAWGraphs, Highcharts)
- An **MCP server** for AI agent integration (Claude, Kiro, Cursor, VS Code Copilot)
- A **REST API server** (`scomp-link serve`) for model deployment

---

## Key Features

| Category | What you get |
|----------|-------------|
| 🎯 **Pipeline** | Automated model selection, training, validation, HTML reports |
| ⌨️ **CLI** | 25 commands: `run`, `predict`, `text`, `cluster`, `tune`, `validate`, `explain`, `engineer`, `forecast`, `anomaly`, `drift`, `fairness`, `quality`, `describe`, `report`, `compare`, `monitor`, `serve`, `export`, `pipeline`, `info`, `init`, `list-models`, `check-deps`, `mcp` |
| 🔧 **Preprocessing** | Polars-backed cleaning, feature engineering (interactions, log, dates, target encoding, binning), data quality profiling |
| 🤖 **Models** | Regression, classification, clustering, time series, anomaly detection, text (BERT contrastive + TF-IDF), images (CNN) |
| ⚙️ **Tuning** | Optuna (Bayesian), Halving Grid Search, Early Stopping CV |
| ✅ **Validation** | K-Fold, LOOCV, Bootstrap, ensemble (voting/stacking) |
| 🔍 **Explainability** | SHAP values, LIME explanations |
| 📊 **Monitoring** | Data drift detection (PSI + KS), anomaly detection, fairness metrics, production monitoring |
| 💾 **Persistence** | Custom `.scomp` format + export to pickle, joblib, ONNX |
| 📈 **Visualization** | 31 RAWGraphs SVG charts, Plotly interactive charts, Highcharts (streamgraph, heatmap, gantt) |
| 📝 **Reporting** | Interactive HTML reports with embedded charts, data quality reports, validation reports |
| 🤝 **Agent Integration** | MCP server (15 tools, 3 resources, 4 prompts) + Agent Skill (SKILL.md) |
| 🚀 **Deployment** | REST API serving (Flask), YAML-driven pipelines |

---

## Quick Start

```bash
pip install scomp-link

# Profile your data
scomp-link describe --data train.csv

# Train a model
scomp-link run --data train.csv --target price --task regression --save-artifact model.scomp

# Tune hyperparameters
scomp-link tune --data train.csv --target price --task regression --method optuna --n-trials 100 --save-artifact best.scomp

# Validate
scomp-link validate --artifact best.scomp --data test.csv --target price --report report.html

# Deploy
scomp-link serve --artifact best.scomp --port 8080
```

---

## Documentation

- [CLI Reference](cli.md) — All 25 commands with examples
- [Python API](API_REFERENCE.md) — Programmatic usage
- [Quick Start](quickstart.md) — Getting started guide
- [Examples](examples.md) — 34 runnable example scripts
- [Visualization Guide](visualization.md) — 39 chart types and HTML reporting
- [Agent Integration](agent-integration.md) — MCP server + SKILL.md for AI agents
- [Changelog](changelog.md) — Version history

---

## Agent Integration

scomp-link works with AI agents out of the box:

```bash
# Start MCP server for Claude/Kiro/Cursor
pip install scomp-link[mcp]
scomp-link mcp

# Or use the Agent Skill (zero deps)
cp -r skills/scomp-link ~/.kiro/skills/
```

See [Agent Integration Guide](agent-integration.md) for setup instructions.
