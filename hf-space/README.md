---
title: scomp-link MCP Server
emoji: 🔧
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
tags:
  - mcp-server
  - machine-learning
  - automl
  - data-science
pinned: false
license: mit
short_description: End-to-end ML toolkit as MCP tool (15 tools)
---

# scomp-link MCP Server

End-to-end ML toolkit exposed as an MCP (Model Context Protocol) server.

## Available Tools (15)

| Tool | Description |
|------|-------------|
| `describe_data` | Profile a dataset (dtypes, missing%, stats) |
| `train_model` | Train a model (regression/classification) |
| `predict` | Predict using a trained artifact |
| `validate_model` | Validate model on test data |
| `detect_drift` | Detect data drift (PSI + KS test) |
| `detect_anomalies` | Anomaly detection (IForest, LOF, TabNet) |
| `check_fairness` | Fairness metrics (demographic parity, disparate impact) |
| `forecast_series` | Time series forecasting |
| `engineer_features` | Feature engineering (interactions, log, dates) |
| `cluster_data` | Clustering (KMeans, DBSCAN, etc.) |
| `generate_report` | Generate HTML EDA/model reports |
| `create_visualization` | Create charts (31 types) |
| `compare_models` | Compare multiple model artifacts |
| `export_model` | Export model to ONNX/PMML/pickle |

## Usage

Connect to this Space from any MCP-compatible client:

**Claude Desktop / Kiro / Cursor:**
```json
{
  "mcpServers": {
    "scomp-link": {
      "url": "https://Euribor512-scomp-link.hf.space/sse"
    }
  }
}
```

## Links

- [PyPI](https://pypi.org/project/scomp-link/)
- [GitHub](https://github.com/GiacomoSaccaggi/scomp_link)
- [Documentation](https://giacomosaccaggi.github.io/scomp_link/)
