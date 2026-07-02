# Command-Line Interface

scomp-link provides a full-featured CLI with **25 commands** for zero-code ML workflows. After installation (`pip install scomp-link`), the `scomp-link` command is available globally.

```bash
scomp-link --help
```

---

## Commands Overview

### Training & Prediction

| Command | Purpose |
|---------|---------|
| [`run`](#run) | Train a model (regression, classification, text, clustering, image) |
| [`predict`](#predict) | Generate predictions from a saved artifact |
| [`text`](#text) | Dedicated text classification (TF-IDF or BERT contrastive) |
| [`cluster`](#cluster) | KMeans or MeanShift clustering |
| [`tune`](#tune) | Hyperparameter tuning (Optuna or HalvingGridSearch) |
| [`pipeline`](#pipeline) | Run multi-step pipeline from YAML config |

### Evaluation & Monitoring

| Command | Purpose |
|---------|---------|
| [`validate`](#validate) | Evaluate a model on test data with metrics + report |
| [`explain`](#explain) | SHAP feature importance |
| [`fairness`](#fairness) | Bias and fairness metrics |
| [`monitor`](#monitor) | Combined drift + quality + performance monitoring |
| [`compare`](#compare) | Side-by-side comparison of multiple artifacts |

### Data & Features

| Command | Purpose |
|---------|---------|
| [`describe`](#describe) | Quick column-level profiling |
| [`quality`](#quality) | Full data quality HTML report |
| [`engineer`](#engineer) | Automated feature engineering |
| [`drift`](#drift) | Distribution drift detection |
| [`anomaly`](#anomaly) | Multi-method anomaly detection |
| [`forecast`](#forecast) | Time series forecasting |

### Model Lifecycle

| Command | Purpose |
|---------|---------|
| [`init`](#init) | Scaffold a new ML project |
| [`serve`](#serve) | Deploy model as REST API |
| [`export`](#export) | Convert .scomp to pickle/ONNX/joblib |
| [`report`](#report) | Generate interactive HTML report |
| [`info`](#info) | Inspect a .scomp artifact |

### Utilities

| Command | Purpose |
|---------|---------|
| [`list-models`](#list-models) | Show available model types |
| [`check-deps`](#check-deps) | Check installed dependencies |
| [`mcp`](#mcp) | Start MCP server for AI agents |

---

## Global Options

```bash
scomp-link --version    # Show version
scomp-link --help       # Show help
```

---

## Training & Prediction

### `run`

Train and evaluate a complete ML pipeline.

```bash
scomp-link run --data train.csv --target price --task regression
scomp-link run --data train.csv --target label --task classification --engineer --ensemble voting
scomp-link run --data train.csv --target label --task text --text-col message
scomp-link run --data train.csv --target label --task clustering --n-clusters 5
```

**Arguments:**

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--data` | path | required | Input data file (.csv, .tsv, .parquet) |
| `--target` | str | required | Target column name |
| `--task` | choice | required | regression, classification, text, clustering, image |
| `--features` | str | all | Comma-separated feature columns |
| `--text-col` | str | - | Text column (for task=text) |
| `--image-col` | str | - | Image column (for task=image) |
| `--n-clusters` | int | - | Number of clusters (for task=clustering) |
| `--model-hint` | str | auto | Force model type |
| `--test-size` | float | 0.2 | Test split ratio |
| `--engineer` | flag | - | Apply feature engineering |
| `--ensemble` | choice | - | voting or stacking |
| `--advanced-cv` | flag | - | LOOCV + Bootstrap CV |
| `--save-artifact` | path | - | Save as .scomp file |
| `--output` | path | - | Results output file |
| `--format` | choice | json | json, csv, or table |
| `--silent` | flag | - | Suppress output |

### `predict`

Generate predictions from a saved .scomp artifact.

```bash
scomp-link predict --artifact model.scomp --data new_data.csv --output predictions.csv
```

### `text`

Dedicated text classification.

```bash
scomp-link text --data tickets.csv --text-col message --target category --method tfidf
scomp-link text --data tickets.csv --text-col message --target category --method contrastive --epochs 5
```

### `cluster`

Clustering with optional visualization.

```bash
scomp-link cluster --data customers.csv --n-clusters 5 --plot clusters.html
scomp-link cluster --data data.csv --method meanshift --output clustered.csv
```

### `tune`

Hyperparameter optimization.

```bash
scomp-link tune --data train.csv --target y --task regression --method optuna --n-trials 100 --save-artifact best.scomp
scomp-link tune --data train.csv --target y --task classification --method halving --format table
```

### `pipeline`

Run a multi-step pipeline from YAML config.

```bash
scomp-link pipeline --config pipeline.yaml
```

See [pipeline template](../skills/scomp-link/assets/pipeline-template.yaml) for config format.

---

## Evaluation & Monitoring

### `validate`

Evaluate a saved artifact on test data.

```bash
scomp-link validate --artifact model.scomp --data test.csv --target y --report validation.html
```

### `explain`

Generate SHAP feature importance.

```bash
scomp-link explain --artifact model.scomp --data test.csv --output importance.csv
```

### `fairness`

Check fairness and bias metrics.

```bash
scomp-link fairness --data predictions.csv --target y_true --predicted y_pred --sensitive gender
```

### `monitor`

Combined drift + quality + performance monitoring.

```bash
scomp-link monitor --reference train.csv --current prod.csv --artifact model.scomp --target y --output monitor.html
```

### `compare`

Compare multiple model artifacts.

```bash
scomp-link compare --artifacts v1.scomp v2.scomp v3.scomp --plot comparison.html
```

---

## Data & Features

### `describe`

Quick dataset profiling (one row per column).

```bash
scomp-link describe --data dataset.csv --format table
scomp-link describe --data dataset.csv --format csv --output profile.csv
```

### `quality`

Full data quality HTML report.

```bash
scomp-link quality --data raw_data.csv --output quality_report.html
```

### `engineer`

Automated feature engineering.

```bash
scomp-link engineer --data raw.csv --target y --interactions --log-transform --output features.csv
```

### `drift`

Distribution drift detection between datasets.

```bash
scomp-link drift --reference train.csv --current production.csv --plot drift.html
```

### `anomaly`

Multi-method consensus anomaly detection.

```bash
scomp-link anomaly --data data.csv --methods iforest,lof,tabnet,transformer --contamination 0.05
```

### `forecast`

Time series forecasting.

```bash
scomp-link forecast --data series.csv --column value --horizon 30 --method auto --plot forecast.html
```

---

## Model Lifecycle

### `init`

Scaffold a new ML project.

```bash
scomp-link init my_project
```

### `serve`

Deploy a model as REST API.

```bash
scomp-link serve --artifact model.scomp --port 8080
```

**Endpoints:**
- `GET /health` — Server status
- `GET /info` — Model metadata
- `GET /schema` — Feature schema
- `POST /predict` — Generate predictions (JSON body: `{"instances": [...]}`)

### `export`

Export model to standard format.

```bash
scomp-link export --artifact model.scomp --format onnx
scomp-link export --artifact model.scomp --format pickle --output model.pkl
```

### `report`

Generate interactive HTML report.

```bash
scomp-link report --data train.csv --output eda_report.html
scomp-link report --artifact model.scomp --data test.csv --output model_report.html
```

### `info`

Inspect a .scomp artifact.

```bash
scomp-link info --artifact model.scomp
```

---

## Utilities

### `list-models`

Show all available model types.

```bash
scomp-link list-models
```

### `check-deps`

Check which optional dependencies are installed.

```bash
scomp-link check-deps
```

### `mcp`

Start the MCP (Model Context Protocol) server for AI agent integration.

```bash
scomp-link mcp
```

Compatible with Claude Desktop, Kiro, Cursor, VS Code Copilot. Exposes 15 tools, 3 resources, and 4 prompts. See [Agent Integration](agent-integration.md) for setup.
