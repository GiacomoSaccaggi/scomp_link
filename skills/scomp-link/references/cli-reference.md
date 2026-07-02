# CLI Reference — All 24 Commands

## Training & Prediction

### `run` — Train a model
```bash
scomp-link run --data PATH --target COL --task {regression,classification,text,clustering,image}
  [--features COL1,COL2]     # Specific features (default: all except target)
  [--text-col COL]           # Text column (for --task text)
  [--image-col COL]          # Image column (for --task image)
  [--n-clusters N]           # Clusters (for --task clustering)
  [--model-hint HINT]        # Force model type (numerical_prediction, categorical_known, etc.)
  [--test-size 0.2]          # Test split ratio
  [--engineer]               # Apply feature engineering before training
  [--ensemble {voting,stacking}]  # Enable ensemble
  [--advanced-cv]            # Run LOOCV + Bootstrap CV
  [--save-artifact PATH]     # Save as .scomp file
  [--output PATH]            # Save results to file
  [--format {json,csv,table}]  # Output format
  [--name NAME]              # Pipeline name
  [--silent]                 # Suppress output
```

### `predict` — Predict with saved artifact
```bash
scomp-link predict --artifact MODEL.scomp --data PATH
  [--output PATH]            # Output file (default: predictions.csv)
  [--silent]
```

### `text` — Text classification
```bash
scomp-link text --data PATH --text-col COL --target COL
  [--method {tfidf,contrastive}]  # tfidf=fast, contrastive=BERT (default: tfidf)
  [--model-name MODEL]       # Transformer model (default: bert-base-uncased)
  [--epochs 3]               # Training epochs
  [--batch-size 32]          # Batch size
  [--test-size 0.2]          # Test split
  [--save-artifact PATH]     # Save artifact
  [--output PATH]            # Results JSON
  [--silent]
```

### `cluster` — Clustering
```bash
scomp-link cluster --data PATH
  [--features COL1,COL2]     # Features to cluster on (default: all numeric)
  [--n-clusters 5]           # Number of clusters
  [--method {kmeans,meanshift}]  # Algorithm (default: kmeans)
  [--output PATH]            # Output CSV with cluster column
  [--plot PATH.html]         # Scatter plot visualization
  [--silent]
```

### `tune` — Hyperparameter tuning
```bash
scomp-link tune --data PATH --target COL --task {regression,classification}
  [--method {optuna,halving}]  # Tuning method (default: optuna)
  [--n-trials 50]            # Number of trials
  [--features COL1,COL2]     # Feature columns
  [--test-size 0.2]          # Test split
  [--save-artifact PATH]     # Save best model
  [--output PATH]            # Results JSON
  [--format {json,csv,table}]
  [--silent]
```

### `pipeline` — Run from YAML config
```bash
scomp-link pipeline --config PATH.yaml
  [--silent]
```

## Evaluation & Monitoring

### `validate` — Evaluate artifact on test data
```bash
scomp-link validate --artifact MODEL.scomp --data PATH --target COL
  [--output PATH]            # Metrics JSON
  [--report PATH.html]       # HTML validation report
  [--format {json,csv,table}]
  [--silent]
```

### `explain` — SHAP feature importance
```bash
scomp-link explain --artifact MODEL.scomp --data PATH
  [--n-samples 100]          # Samples to explain
  [--output PATH]            # Feature importance CSV
  [--silent]
```

### `fairness` — Bias and fairness metrics
```bash
scomp-link fairness --data PATH --target COL --predicted COL --sensitive COL
  [--output PATH]            # Report JSON
  [--silent]
```

### `monitor` — Production monitoring report
```bash
scomp-link monitor --reference PATH --current PATH
  [--artifact MODEL.scomp]   # For performance metrics
  [--target COL]             # Target column (with --artifact)
  [--threshold 0.2]          # PSI drift threshold
  [--output PATH.html]       # Output report
  [--silent]
```

### `compare` — Compare multiple artifacts
```bash
scomp-link compare --artifacts A.scomp B.scomp [C.scomp ...]
  [--output PATH]            # Comparison CSV
  [--plot PATH.html]         # Bar chart comparison
```

## Data & Features

### `describe` — Quick dataset profiling
```bash
scomp-link describe --data PATH
  [--format {table,csv,json}]  # Output format (default: table)
  [--output PATH]            # Save to file
```
Output: one row per column with dtype, missing%, unique, min, max, mean, std.

### `quality` — Full data quality report
```bash
scomp-link quality --data PATH
  [--output PATH.html]       # HTML report (default: data_quality_report.html)
  [--silent]
```

### `engineer` — Feature engineering
```bash
scomp-link engineer --data PATH
  [--target COL]             # Target for target encoding
  [--interactions]           # Polynomial interactions
  [--log-transform]          # Log-transform skewed features
  [--date-features]          # Extract year/month/dow from dates
  [--target-encode]          # Target encode high-cardinality categoricals
  [--auto-bin]               # Quantile binning
  [--n-bins 5]               # Number of bins
  [--output PATH]            # Output file (default: engineered.csv)
  [--silent]
```

### `drift` — Distribution drift detection
```bash
scomp-link drift --reference PATH --current PATH
  [--features COL1,COL2]     # Specific features (default: all numeric)
  [--threshold 0.2]          # PSI threshold
  [--output PATH]            # Drift report CSV
  [--plot PATH.html]         # PSI bar chart
  [--silent]
```

### `anomaly` — Anomaly detection
```bash
scomp-link anomaly --data PATH
  [--features COL1,COL2]     # Features (default: all numeric)
  [--methods iforest,lof,tabnet,transformer]  # Detection methods
  [--contamination 0.05]     # Expected anomaly fraction
  [--consensus 2]            # Min methods that must agree
  [--output PATH]            # Output CSV with anomaly labels
  [--plot PATH.html]         # Anomaly visualization
  [--silent]
```

### `forecast` — Time series forecasting
```bash
scomp-link forecast --data PATH --column COL
  [--horizon 10]             # Steps to forecast
  [--method {auto,arima,sarima,exp_smoothing}]
  [--seasonal-period N]      # Seasonal period
  [--cv-splits N]            # Walk-forward CV splits
  [--output PATH]            # Forecast CSV
  [--plot PATH.html]         # Forecast chart
  [--silent]
```

## Model Lifecycle

### `init` — Scaffold new project
```bash
scomp-link init NAME
  [--force]                  # Overwrite existing
```
Creates: `NAME/{data/, models/, reports/, pipeline.py, config.yaml, .gitignore, README.md}`

### `serve` — REST API server
```bash
scomp-link serve --artifact MODEL.scomp
  [--host 0.0.0.0]           # Bind address
  [--port 8080]              # Port
  [--debug]                  # Flask debug mode
```
Endpoints: `GET /health`, `GET /info`, `GET /schema`, `POST /predict`

POST /predict body: `{"instances": [{"col1": val1, "col2": val2}]}`

### `export` — Export to standard format
```bash
scomp-link export --artifact MODEL.scomp
  [--format {pickle,joblib,onnx,pmml}]  # Export format (default: pickle)
  [--output PATH]            # Output file
```

### `report` — Interactive HTML report
```bash
# EDA report
scomp-link report --data PATH --output PATH.html

# Model evaluation report (requires artifact + data)
scomp-link report --artifact MODEL.scomp --data PATH --output PATH.html
  [--silent]
```

### `info` — Inspect artifact metadata
```bash
scomp-link info --artifact MODEL.scomp
```
Output: JSON with model_type, task, target, metrics, feature_schema, metadata.

## Utilities

### `list-models` — Available model types
```bash
scomp-link list-models
```

### `check-deps` — Dependency status
```bash
scomp-link check-deps
```

### `mcp` — Start MCP server
```bash
scomp-link mcp
```
Starts the Model Context Protocol server for agent integration (stdio transport).
