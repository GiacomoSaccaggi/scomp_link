# Command-Line Interface

scomp-link provides a full-featured CLI for zero-code ML workflows. After installation (`pip install scomp-link`), the `scomp-link` command is available globally.

```bash
scomp-link --help
```

---

## Commands Overview

| Command | Purpose |
|---------|---------|
| [`run`](#run) | Train and evaluate a complete ML pipeline |
| [`predict`](#predict) | Generate predictions from a saved artifact |
| [`explain`](#explain) | Compute SHAP feature importance |
| [`engineer`](#engineer) | Apply automated feature engineering |
| [`forecast`](#forecast) | Time series forecasting |
| [`anomaly`](#anomaly) | Multi-method anomaly detection |
| [`drift`](#drift) | Detect data drift between datasets |
| [`fairness`](#fairness) | Check fairness and bias metrics |
| [`quality`](#quality) | Generate data quality report |
| [`compare`](#compare) | Compare multiple model artifacts |
| [`report`](#report) | Generate interactive HTML report (EDA or model evaluation) |
| [`info`](#info) | Inspect a `.scomp` artifact |

---

## Global Options

| Flag | Description |
|------|-------------|
| `--version` | Show version and exit |
| `--help` | Show help message |

---

## `run`

Train and evaluate a model on tabular data. Supports automatic model selection, feature engineering, ensemble learning, and advanced cross-validation.

```bash
scomp-link run --data train.csv --target price --task regression
```

### Options

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--data` | ✅ | — | Path to input data (CSV, TSV, Parquet) |
| `--target` | ✅ | — | Target column name |
| `--task` | ✅ | — | `regression` or `classification` |
| `--features` | | all | Comma-separated feature columns |
| `--model-hint` | | auto | Model selection hint (e.g. `numerical_prediction`) |
| `--test-size` | | 0.2 | Test split ratio |
| `--name` | | filename | Pipeline name |
| `--output`, `-o` | | stdout | Output path for results JSON |
| `--save-artifact` | | — | Save pipeline as `.scomp` file |
| `--engineer` | | false | Apply feature engineering before training |
| `--ensemble` | | — | Enable ensemble: `voting` or `stacking` |
| `--advanced-cv` | | false | Run LOOCV + Bootstrap validation |
| `--silent` | | false | Suppress progress output |

### Examples

```bash
# Basic regression
scomp-link run --data housing.csv --target price --task regression

# Classification with ensemble and artifact save
scomp-link run --data customers.csv --target churn --task classification \
  --ensemble voting --save-artifact churn_model.scomp --output results.json

# With feature engineering and advanced CV
scomp-link run --data data.parquet --target y --task regression \
  --engineer --advanced-cv --silent
```

### Output (JSON)

```json
{
  "status": "success",
  "model_type": "Lasso / Elastic Net",
  "metrics": {"mse": 0.42, "rmse": 0.65, "mae": 0.51, "r2": 0.87},
  "report_path": "ScompLink_Validation_Report.html"
}
```

---

## `predict`

Load a saved `.scomp` artifact and generate predictions on new data.

```bash
scomp-link predict --artifact model.scomp --data new_data.csv --output predictions.csv
```

### Options

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--artifact` | ✅ | — | Path to `.scomp` artifact |
| `--data` | ✅ | — | Path to input data |
| `--output`, `-o` | | `predictions.csv` | Output path |
| `--silent` | | false | Suppress output |

The output file contains all input columns plus a `prediction` column.

---

## `explain`

Compute SHAP values and output feature importance rankings.

```bash
scomp-link explain --artifact model.scomp --data test.csv --output importance.csv
```

### Options

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--artifact` | ✅ | — | Path to `.scomp` artifact |
| `--data` | ✅ | — | Data to explain |
| `--n-samples` | | 100 | Number of samples to compute SHAP on |
| `--output`, `-o` | | `feature_importance.csv` | Output path |
| `--silent` | | false | Suppress output |

---

## `engineer`

Apply automated feature engineering transformations to a dataset.

```bash
scomp-link engineer --data raw.csv --target y --interactions --log-transform --output engineered.parquet
```

### Options

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--data` | ✅ | — | Path to input data |
| `--target` | | — | Target column (needed for target encoding) |
| `--output`, `-o` | | `engineered.csv` | Output path |
| `--interactions` | | false | Generate polynomial interaction features |
| `--log-transform` | | false | Log-transform skewed features |
| `--date-features` | | false | Extract date components (year, month, dow, etc.) |
| `--target-encode` | | false | Target-encode high-cardinality categoricals |
| `--auto-bin` | | false | Bin continuous features into quantile buckets |
| `--n-bins` | | 5 | Number of bins |
| `--silent` | | false | Suppress output |

### Transformations Applied

- **Interactions**: creates `col_a_x_col_b` for top numeric pairs
- **Log transform**: applies `log1p` to features with |skewness| > 1.0 (non-negative only)
- **Date features**: extracts `_year`, `_month`, `_day_of_week`, `_is_weekend`, `_quarter`
- **Target encoding**: replaces high-cardinality categoricals with mean(target) per category
- **Auto-binning**: creates `_bin` columns with quantile-based bucket labels

---

## `forecast`

Forecast future values from a time series.

```bash
scomp-link forecast --data series.csv --column value --horizon 30 --method arima
```

### Options

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--data` | ✅ | — | Path to input data |
| `--column` | ✅ | — | Column containing time series values |
| `--horizon` | | 10 | Number of steps to forecast |
| `--method` | | `auto` | `auto`, `arima`, `sarima`, `exp_smoothing` |
| `--seasonal-period` | | auto-detect | Seasonal period (e.g. 12 for monthly) |
| `--cv-splits` | | — | Walk-forward CV splits (optional) |
| `--output`, `-o` | | `forecast.csv` | Output path |
| `--silent` | | false | Suppress output |

### Output Columns

| Column | Description |
|--------|-------------|
| `step` | Future time index |
| `forecast` | Predicted value |
| `lower` | Lower 95% confidence bound |
| `upper` | Upper 95% confidence bound |

---

## `anomaly`

Detect anomalies using multi-method consensus voting.

```bash
scomp-link anomaly --data data.csv --methods iforest,lof --contamination 0.05
```

### Options

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--data` | ✅ | — | Path to input data |
| `--features` | | all numeric | Comma-separated feature columns |
| `--methods` | | `iforest,lof` | Methods: `iforest`, `lof`, `tabnet`, `transformer` |
| `--contamination` | | 0.05 | Expected anomaly fraction |
| `--consensus` | | 2 | Minimum methods that must agree |
| `--output`, `-o` | | `anomalies.csv` | Output path |
| `--silent` | | false | Suppress output |

Output includes all original columns plus `is_anomaly` (boolean) and per-method flags.

---

## `drift`

Compare data distributions between reference and current datasets using PSI and KS tests.

```bash
scomp-link drift --reference train.csv --current production.csv --threshold 0.2
```

### Options

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--reference` | ✅ | — | Reference (training) data |
| `--current` | ✅ | — | Current (production) data |
| `--features` | | all numeric | Comma-separated features to check |
| `--threshold` | | 0.2 | PSI threshold for drift |
| `--output`, `-o` | | stdout | Output path for CSV report |
| `--silent` | | false | Suppress output |

### Output Columns

| Column | Description |
|--------|-------------|
| `feature` | Feature name |
| `psi` | Population Stability Index |
| `ks_statistic` | KS test statistic |
| `p_value` | KS test p-value |
| `drifted` | Boolean: drift detected |

---

## `fairness`

Evaluate fairness metrics on binary classification predictions.

```bash
scomp-link fairness --data preds.csv --target y_true --predicted y_pred --sensitive gender
```

### Options

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--data` | ✅ | — | Data with true labels, predictions, and protected attribute |
| `--target` | ✅ | — | True label column (binary 0/1) |
| `--predicted` | ✅ | — | Predicted label column (binary 0/1) |
| `--sensitive` | ✅ | — | Protected attribute column |
| `--output`, `-o` | | stdout | Output path for JSON report |
| `--silent` | | false | Suppress output |

### Metrics Computed

| Metric | Fair if |
|--------|---------|
| Demographic Parity ratio | ≥ 0.8 |
| Disparate Impact (4/5 rule) | ≥ 0.8 |
| Equal Opportunity (TPR diff) | < 0.1 |
| Equalized Odds (TPR + FPR diff) | < 0.1 |

---

## `quality`

Profile a dataset and generate a standalone HTML report.

```bash
scomp-link quality --data raw_data.csv --output report.html
```

### Options

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--data` | ✅ | — | Path to data to profile |
| `--output`, `-o` | | `data_quality_report.html` | Output path (.html or .csv) |
| `--silent` | | false | Suppress output |

### Report Includes

- Dataset overview (rows, columns, memory, types)
- Missing values analysis
- Cardinality profiling with flags (near-unique, binary, etc.)
- Constant feature detection
- Duplicate row count
- High correlation pairs (≥ 0.95)

---

## `compare`

Side-by-side comparison of multiple `.scomp` model artifacts.

```bash
scomp-link compare --artifacts model_v1.scomp model_v2.scomp model_v3.scomp
```

### Options

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--artifacts` | ✅ | — | Paths to `.scomp` files (space-separated) |
| `--output`, `-o` | | stdout | Output path (CSV) |

### Output

A table with one row per artifact showing model type and all stored metrics.

---

## `report`

Generate an interactive HTML report — either an EDA report on a dataset, or a model evaluation report from a `.scomp` artifact.

### EDA Report (dataset profiling)

```bash
scomp-link report --data train.csv --output eda_report.html
```

Generates:

- Dataset overview (rows, columns, types, missing count)
- Histograms for all numeric features (up to 12)
- Correlation heatmap
- Missing values bar chart

### Model Report (artifact evaluation)

```bash
scomp-link report --artifact model.scomp --data test.csv --output model_report.html
```

Generates:

- Model info (type, task, target, n_features)
- Metrics table
- SHAP feature importance chart
- Predicted vs Actual scatter plot
- Residual distribution histogram

### Options

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--data` | for EDA | — | Path to data |
| `--artifact` | for model report | — | Path to `.scomp` artifact (triggers model mode) |
| `--output`, `-o` | | `report.html` | Output HTML path |
| `--silent` | | false | Suppress progress output |

### Examples

```bash
# Quick EDA on any CSV
scomp-link report --data raw_data.csv

# Full model evaluation
scomp-link report --artifact production_model.scomp --data test_set.parquet --output eval.html

# Model report without test data (just metrics + info)
scomp-link report --artifact model.scomp
```

---

## `info`

Show metadata, configuration, and metrics of a saved `.scomp` pipeline artifact.

```bash
scomp-link info --artifact model.scomp
```

### Options

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `--artifact` | ✅ | — | Path to `.scomp` file |

### Output (JSON)

```json
{
  "config": {"task_type": "regression", "target_col": "price"},
  "metrics": {"r2": 0.87, "rmse": 0.65},
  "n_features": 12,
  "has_model": true,
  "has_preprocessor": true,
  "has_sample_data": true,
  "model_type": "GradientBoostingRegressor"
}
```

---

## Supported File Formats

| Extension | Read | Write |
|-----------|------|-------|
| `.csv` | ✅ | ✅ |
| `.tsv` | ✅ | — |
| `.parquet` | ✅ | ✅ |
| `.json` | — | ✅ |
| `.html` | — | ✅ (quality report) |

---

## `init`

Scaffold a new ML project with a ready-to-use directory structure.

```bash
scomp-link init my_project
```

### Options

| Flag | Required | Default | Description |
|------|----------|---------|-------------|
| `name` | ✅ (positional) | — | Project name (creates directory) |
| `--force` | | false | Overwrite existing directory |

### Generated Structure

```
my_project/
├── data/          # Place input datasets here
├── models/        # Saved .scomp artifacts
├── reports/       # Generated HTML reports
├── pipeline.py    # Main pipeline script (edit target/task)
├── config.yaml    # Project configuration
├── .gitignore     # Pre-configured ignores
└── README.md      # Project documentation
```

The generated `pipeline.py` is a working template — just add your data and change the target column.

---

## Typical Workflow

```bash
# 0. Scaffold a new project
scomp-link init my_project

# 1. Profile your data
scomp-link quality --data raw_data.csv --output quality_report.html

# 2. Engineer features
scomp-link engineer --data raw_data.csv --target price \
  --interactions --log-transform --date-features --output features.parquet

# 3. Train model
scomp-link run --data features.parquet --target price --task regression \
  --ensemble voting --save-artifact model.scomp --output results.json

# 4. Inspect model
scomp-link info --artifact model.scomp

# 5. Explain model
scomp-link explain --artifact model.scomp --data features.parquet

# 6. Predict on new data
scomp-link predict --artifact model.scomp --data new_data.parquet --output predictions.csv

# 7. Monitor drift in production
scomp-link drift --reference features.parquet --current production_data.csv

# 8. Check fairness
scomp-link fairness --data predictions.csv --target y_true --predicted prediction --sensitive gender
```
