# -*- coding: utf-8 -*-
"""
 ██████╗██╗     ██╗
██╔════╝██║     ██║
██║     ██║     ██║
██║     ██║     ██║
╚██████╗███████╗██║
 ╚═════╝╚══════╝╚═╝

scomp-link CLI — zero-code ML pipeline from the terminal.
"""
import argparse
import sys
import json
from pathlib import Path
from typing import Optional


def _load_data(path: str, target: Optional[str] = None):
    """Load data from CSV/Parquet file."""
    import pandas as pd
    p = Path(path)
    if not p.exists():
        sys.exit(f"Error: file not found: {path}")
    if p.suffix == '.parquet':
        df = pd.read_parquet(p)
    elif p.suffix in ('.csv', '.tsv'):
        sep = '\t' if p.suffix == '.tsv' else ','
        df = pd.read_csv(p, sep=sep)
    else:
        sys.exit(f"Error: unsupported file format: {p.suffix} (use .csv, .tsv, or .parquet)")
    if target and target not in df.columns:
        sys.exit(f"Error: target column '{target}' not found. Available: {list(df.columns)}")
    return df


def _write_output(data, path: str, fmt: str = "auto"):
    """Write output to file (CSV, Parquet, or JSON)."""
    import pandas as pd
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "auto":
        fmt = p.suffix.lstrip('.')

    if fmt == "parquet":
        if isinstance(data, pd.DataFrame):
            data.to_parquet(p, index=False)
        else:
            pd.DataFrame(data).to_parquet(p, index=False)
    elif fmt == "csv":
        if isinstance(data, pd.DataFrame):
            data.to_csv(p, index=False)
        else:
            pd.DataFrame(data).to_csv(p, index=False)
    elif fmt == "json":
        with open(p, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    else:
        sys.exit(f"Error: unsupported output format: {fmt} (use csv, parquet, json)")
    return p


# ═══════════════════════════════════════════════════════════════════
# SUBCOMMANDS
# ═══════════════════════════════════════════════════════════════════

def cmd_run(args):
    """Run a complete ML pipeline on a dataset."""
    import scomp_link
    if args.silent:
        scomp_link.set_verbosity("silent")

    df = _load_data(args.data, args.target)

    # Optional: apply feature engineering
    if args.engineer:
        from scomp_link import FeatureEngineer
        y_tmp = df[args.target]
        X_tmp = df.drop(columns=[args.target])
        fe = FeatureEngineer(interactions=True, log_transform=True, date_features=True,
                             target_encode=True, auto_bin=False)
        X_tmp = fe.fit_transform(X_tmp, y_tmp)
        X_tmp[args.target] = y_tmp.values
        df = X_tmp

    pipe = scomp_link.ScompLinkPipeline(args.name or Path(args.data).stem)
    pipe.set_objectives([f"Optimize {args.task}"])
    pipe.import_and_clean_data(df)

    feature_cols = args.features.split(',') if args.features else None
    pipe.select_variables(target_col=args.target, feature_cols=feature_cols)

    metadata = {}
    if args.model_hint:
        pipe.choose_model(args.model_hint, metadata=metadata)
    else:
        pipe.choose_model("numerical_prediction" if args.task == "regression" else "categorical_known",
                          metadata=metadata)

    kwargs = dict(task_type=args.task, test_size=args.test_size)
    if args.ensemble:
        kwargs['use_ensemble'] = True
        kwargs['ensemble_strategy'] = args.ensemble
    if args.advanced_cv:
        kwargs['advanced_cv'] = True

    results = pipe.run_pipeline(**kwargs)

    # Output
    output = {
        "status": results.get("status"),
        "model_type": results.get("model_type"),
        "metrics": results.get("metrics"),
        "report_path": results.get("report_path"),
    }

    if args.output:
        _write_output(output, args.output, fmt="json")
        print(f"Results saved to: {args.output}")
    else:
        print(json.dumps(output, indent=2, default=str))

    # Save artifact if requested
    if args.save_artifact:
        artifact = scomp_link.ScompArtifact()
        artifact.set_model(pipe.model)
        artifact.set_config(task_type=args.task, target_col=args.target)
        artifact.set_metrics(results.get("metrics", {}))
        artifact.set_feature_schema(df.drop(columns=[args.target]))
        artifact.set_sample_data(df.drop(columns=[args.target]))
        artifact.save(args.save_artifact)


def cmd_predict(args):
    """Load a .scomp artifact and predict on new data."""
    import scomp_link
    if args.silent:
        scomp_link.set_verbosity("silent")

    artifact_path = Path(args.artifact)
    if not artifact_path.exists():
        sys.exit(f"Error: artifact not found: {args.artifact}")

    artifact = scomp_link.ScompArtifact.load(args.artifact)
    df = _load_data(args.data)

    # Drop target column if present
    target_col = artifact.config.get('target_col')
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]

    predictions = artifact.predict(X)

    import pandas as pd
    out_df = df.copy()
    out_df['prediction'] = predictions

    output_path = args.output or "predictions.csv"
    _write_output(out_df, output_path)
    print(f"Predictions saved to: {output_path} ({len(predictions)} rows)")


def cmd_explain(args):
    """Generate SHAP explanations for a model artifact."""
    import scomp_link
    if args.silent:
        scomp_link.set_verbosity("silent")

    artifact_path = Path(args.artifact)
    if not artifact_path.exists():
        sys.exit(f"Error: artifact not found: {args.artifact}")

    artifact = scomp_link.ScompArtifact.load(args.artifact)
    df = _load_data(args.data)

    # Drop target column if present
    target_col = artifact.config.get('target_col')
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]

    if artifact.preprocessor is not None:
        import pandas as pd
        X = pd.DataFrame(artifact.preprocessor.transform(X), columns=feature_cols)

    explainer = scomp_link.ShapExplainer(artifact.model, X[:min(100, len(X))])
    explainer.explain(X[:min(args.n_samples, len(X))])
    importance = explainer.feature_importance()

    output_path = args.output or "feature_importance.csv"
    _write_output(importance, output_path)
    print(f"Feature importance saved to: {output_path}")
    print(importance.head(10).to_string(index=False))


def cmd_drift(args):
    """Detect data drift between reference and current data."""
    import scomp_link
    if args.silent:
        scomp_link.set_verbosity("silent")

    reference = _load_data(args.reference)
    current = _load_data(args.current)

    features = args.features.split(',') if args.features else None

    detector = scomp_link.DriftDetector(reference, psi_threshold=args.threshold)
    report = detector.detect(current, features=features)
    summary = detector.summary(report)

    if args.output:
        _write_output(report, args.output)
        print(f"Drift report saved to: {args.output}")
    else:
        print(report.to_string(index=False))

    print(f"\nSummary: {summary['drifted_features']}/{summary['total_features']} features drifted "
          f"(worst: {summary['worst_feature']}, PSI={summary['max_psi']:.4f})")


def cmd_quality(args):
    """Generate a data quality report."""
    import scomp_link
    if args.silent:
        scomp_link.set_verbosity("silent")

    df = _load_data(args.data)
    dqr = scomp_link.DataQualityReport(df)
    report = dqr.generate()

    output_path = args.output or "data_quality_report.html"
    if output_path.endswith('.html'):
        dqr.save_html(output_path)
    else:
        _write_output(report['cardinality'], output_path)
    print(f"Report saved to: {output_path}")


def cmd_report(args):
    """Generate an interactive HTML report (EDA or model evaluation)."""
    import scomp_link
    import pandas as pd
    if args.silent:
        scomp_link.set_verbosity("silent")

    from scomp_link.utils.report_html import ScompLinkHTMLReport
    from scomp_link.utils.plotly_utils import histogram, barchart, linechart

    output_path = args.output or "report.html"

    if args.artifact:
        # Model report: metrics + feature importance + predictions vs actual
        artifact_path = Path(args.artifact)
        if not artifact_path.exists():
            sys.exit(f"Error: artifact not found: {args.artifact}")

        scomp_link.set_verbosity("silent")
        artifact = scomp_link.ScompArtifact.load(args.artifact)
        if args.silent:
            scomp_link.set_verbosity("silent")
        else:
            scomp_link.set_verbosity("info")

        report = ScompLinkHTMLReport(title="Model Report")
        report.open_section("Model Info")
        info_df = pd.DataFrame([{
            "model_type": type(artifact.model).__name__,
            "task": artifact.config.get("task_type", "unknown"),
            "target": artifact.config.get("target_col", "unknown"),
            "n_features": len(artifact.feature_schema),
        }])
        report.add_dataframe(info_df, "Pipeline Configuration")

        if artifact.metrics:
            metrics_df = pd.DataFrame([artifact.metrics])
            report.add_dataframe(metrics_df, "Metrics")
        report.close_section()

        if args.data:
            df = _load_data(args.data)
            target_col = artifact.config.get("target_col")
            feature_cols = [c for c in df.columns if c != target_col]
            X = df[feature_cols]

            # Feature importance via SHAP
            report.open_section("Feature Importance")
            try:
                explainer = scomp_link.ShapExplainer(artifact.model, X[:min(100, len(X))])
                explainer.explain(X[:min(50, len(X))])
                fig = explainer.plot_importance(top_n=min(15, len(feature_cols)))
                report.add_graph_to_report(fig, "SHAP Feature Importance")
            except Exception:
                report.add_title("SHAP not available for this model type")
            report.close_section()

            # Predictions vs Actual
            if target_col and target_col in df.columns:
                report.open_section("Predictions vs Actual")
                predictions = artifact.predict(X)
                import plotly.express as px
                fig_scatter = px.scatter(x=df[target_col], y=predictions,
                                         labels={"x": "Actual", "y": "Predicted"},
                                         title="Predictions vs Actual")
                fig_scatter.add_shape(type="line", x0=df[target_col].min(), x1=df[target_col].max(),
                                      y0=df[target_col].min(), y1=df[target_col].max(),
                                      line=dict(dash="dash", color="red"))
                report.add_graph_to_report(fig_scatter, "Scatter: Predicted vs Actual")

                residuals = df[target_col].values - predictions
                fig_res = histogram(residuals, "Residual Distribution")
                report.add_graph_to_report(fig_res, "Residuals")
                report.close_section()

        report.save_html(output_path)

    else:
        # EDA report on a dataset
        if not args.data:
            sys.exit("Error: --data is required for EDA report")
        df = _load_data(args.data)

        report = ScompLinkHTMLReport(title="EDA Report")

        # Overview
        report.open_section("Dataset Overview")
        overview_df = pd.DataFrame([{
            "rows": len(df), "columns": len(df.columns),
            "numeric": len(df.select_dtypes(include=['number']).columns),
            "categorical": len(df.select_dtypes(include=['object', 'category']).columns),
            "missing_total": int(df.isnull().sum().sum()),
        }])
        report.add_dataframe(overview_df, "Overview")
        report.close_section()

        # Distributions
        import plotly.express as px
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

        report.open_section("Feature Distributions")
        for col in numeric_cols[:12]:  # cap at 12
            fig = histogram(df[col].dropna().values, f"Distribution: {col}")
            report.add_graph_to_report(fig, col)
        report.close_section()

        # Correlations
        if len(numeric_cols) >= 2:
            report.open_section("Correlations")
            corr = df[numeric_cols].corr()
            fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                                  title="Correlation Matrix")
            report.add_graph_to_report(fig_corr, "Correlation Matrix")
            report.close_section()

        # Missing values
        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if len(missing) > 0:
            report.open_section("Missing Values")
            fig_miss = barchart(missing.index.tolist(), missing.values.tolist(), "Missing Values per Column")
            report.add_graph_to_report(fig_miss, "Missing Values")
            report.close_section()

        report.save_html(output_path)

    print(f"Report saved to: {output_path}")


def cmd_info(args):
    """Inspect a .scomp artifact."""
    import scomp_link

    if not scomp_link.ScompArtifact.is_scomp_file(args.artifact):
        sys.exit(f"Error: not a valid .scomp file: {args.artifact}")

    scomp_link.set_verbosity("silent")
    artifact = scomp_link.ScompArtifact.load(args.artifact)
    info = artifact.info()

    print(json.dumps(info, indent=2, default=str))


def cmd_engineer(args):
    """Apply feature engineering to a dataset."""
    import scomp_link
    if args.silent:
        scomp_link.set_verbosity("silent")

    df = _load_data(args.data, args.target)

    y = df[args.target] if args.target else None
    X = df.drop(columns=[args.target]) if args.target else df

    fe = scomp_link.FeatureEngineer(
        interactions=args.interactions,
        log_transform=args.log_transform,
        date_features=args.date_features,
        target_encode=args.target_encode,
        auto_bin=args.auto_bin,
        n_bins=args.n_bins,
    )
    X_eng = fe.fit_transform(X, y)
    if args.target:
        X_eng[args.target] = y.values

    output_path = args.output or "engineered.csv"
    _write_output(X_eng, output_path)
    print(f"Engineered data saved to: {output_path} (shape: {X_eng.shape})")


def cmd_forecast(args):
    """Run time series forecasting."""
    import scomp_link
    import pandas as pd
    if args.silent:
        scomp_link.set_verbosity("silent")

    df = _load_data(args.data)
    if args.column not in df.columns:
        sys.exit(f"Error: column '{args.column}' not found. Available: {list(df.columns)}")

    series = df[args.column].dropna()
    fc = scomp_link.TimeSeriesForecaster(
        method=args.method, horizon=args.horizon,
        seasonal_period=args.seasonal_period,
    )

    if args.cv_splits:
        cv_results = fc.walk_forward_cv(series, n_splits=args.cv_splits, horizon=args.horizon)
        print(f"Walk-forward CV: MAE={cv_results['mean_mae']:.4f}, RMSE={cv_results['mean_rmse']:.4f}")

    fc.fit(series)
    ci = fc.predict_with_ci(steps=args.horizon)
    ci.index = range(len(series), len(series) + args.horizon)
    ci.index.name = 'step'

    output_path = args.output or "forecast.csv"
    _write_output(ci.reset_index(), output_path)
    print(f"Forecast saved to: {output_path} ({args.horizon} steps)")


def cmd_anomaly(args):
    """Detect anomalies in tabular data."""
    import scomp_link
    if args.silent:
        scomp_link.set_verbosity("silent")

    df = _load_data(args.data)
    features = args.features.split(',') if args.features else df.select_dtypes(include=['number']).columns.tolist()

    methods = args.methods.split(',') if args.methods else ['iforest', 'lof']
    detector = scomp_link.AnomalyDetector(
        contamination=args.contamination,
        methods=methods,
        consensus_threshold=args.consensus,
    )
    results = detector.fit_predict(df, features=features)

    output_path = args.output or "anomalies.csv"
    _write_output(results['data'], output_path)
    print(f"Anomaly detection saved to: {output_path}")
    print(results['comparison'].to_string(index=False))


def cmd_fairness(args):
    """Check fairness metrics on predictions."""
    import scomp_link
    if args.silent:
        scomp_link.set_verbosity("silent")

    df = _load_data(args.data)
    for col in [args.target, args.predicted, args.sensitive]:
        if col not in df.columns:
            sys.exit(f"Error: column '{col}' not found. Available: {list(df.columns)}")

    fm = scomp_link.FairnessMetrics(
        df[args.target].values, df[args.predicted].values,
        sensitive_feature=df[args.sensitive].values
    )
    report = fm.compute_all()
    summary = fm.summary(report)

    if args.output:
        _write_output(report, args.output, fmt="json")
        print(f"Fairness report saved to: {args.output}")
    else:
        print(summary.to_string(index=False))


def cmd_compare(args):
    """Compare multiple .scomp artifacts."""
    import scomp_link
    import pandas as pd

    rows = []
    for artifact_path in args.artifacts:
        if not Path(artifact_path).exists():
            sys.exit(f"Error: artifact not found: {artifact_path}")
        scomp_link.set_verbosity("silent")
        a = scomp_link.ScompArtifact.load(artifact_path)
        row = {"artifact": Path(artifact_path).name, "model_type": type(a.model).__name__}
        row.update(a.metrics)
        rows.append(row)

    comparison = pd.DataFrame(rows)

    if args.output:
        _write_output(comparison, args.output)
        print(f"Comparison saved to: {args.output}")
    else:
        print(comparison.to_string(index=False))


def cmd_init(args):
    """Scaffold a new scomp-link project."""
    project_dir = Path(args.name)
    if project_dir.exists() and not args.force:
        sys.exit(f"Error: directory '{args.name}' already exists. Use --force to overwrite.")

    project_dir.mkdir(parents=True, exist_ok=True)
    (project_dir / "data").mkdir(exist_ok=True)
    (project_dir / "models").mkdir(exist_ok=True)
    (project_dir / "reports").mkdir(exist_ok=True)

    # Main pipeline script
    (project_dir / "pipeline.py").write_text(f'''# -*- coding: utf-8 -*-
"""
{args.name} — scomp-link ML pipeline
Generated by: scomp-link init
"""
import pandas as pd
from scomp_link import ScompLinkPipeline, ScompArtifact, set_verbosity

# set_verbosity("silent")  # uncomment to suppress output

# --- Load data ---
df = pd.read_csv("data/dataset.csv")

# --- Build pipeline ---
pipe = ScompLinkPipeline("{args.name}")
pipe.set_objectives(["Minimize RMSE"])
pipe.import_and_clean_data(df)
pipe.select_variables(target_col="target")  # <-- change to your target column
pipe.choose_model("numerical_prediction")

# --- Run ---
results = pipe.run_pipeline(task_type="regression", test_size=0.2)
print(results)

# --- Save artifact ---
artifact = ScompArtifact()
artifact.set_model(pipe.model)
artifact.set_config(task_type="regression", target_col="target")
artifact.set_metrics(results.get("metrics", {{}}))
artifact.save("models/{args.name}.scomp")
''')

    # Config file
    (project_dir / "config.yaml").write_text(f'''# {args.name} configuration
project_name: "{args.name}"
task_type: regression  # regression | classification
target_column: target
test_size: 0.2
features: []  # empty = use all columns except target

# Feature engineering
engineer:
  interactions: false
  log_transform: true
  date_features: false
  target_encode: false

# Model selection
model_hint: numerical_prediction  # numerical_prediction | categorical_known | categorical_unknown
ensemble: null  # voting | stacking | null
advanced_cv: false
''')

    # .gitignore
    (project_dir / ".gitignore").write_text('''__pycache__/
*.py[cod]
.venv/
*.scomp
*.html
data/*.csv
data/*.parquet
models/
reports/
''')

    # README
    (project_dir / "README.md").write_text(f'''# {args.name}

ML project scaffolded with [scomp-link](https://pypi.org/project/scomp-link/).

## Quick Start

```bash
# 1. Add your data
cp your_data.csv data/dataset.csv

# 2. Edit pipeline.py (set target column, task type)

# 3. Run
python pipeline.py

# Or use the CLI
scomp-link run --data data/dataset.csv --target target --task regression --save-artifact models/{args.name}.scomp
```

## Project Structure

```
{args.name}/
├── data/          # Input datasets
├── models/        # Saved .scomp artifacts
├── reports/       # Generated HTML reports
├── pipeline.py    # Main pipeline script
├── config.yaml    # Project configuration
└── README.md
```
''')

    print(f"✅ Project '{args.name}' created with structure:")
    print(f"   {args.name}/")
    print(f"   ├── data/")
    print(f"   ├── models/")
    print(f"   ├── reports/")
    print(f"   ├── pipeline.py")
    print(f"   ├── config.yaml")
    print(f"   ├── .gitignore")
    print(f"   └── README.md")
    print(f"\n   Next: add data to data/, edit pipeline.py, and run it.")



# ═══════════════════════════════════════════════════════════════════
# PARSER
# ═══════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scomp-link",
        description="scomp-link: End-to-end ML toolkit CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  scomp-link run --data train.csv --target price --task regression
  scomp-link run --data data.csv --target label --task classification --engineer --ensemble voting
  scomp-link predict --artifact model.scomp --data new_data.csv --output predictions.csv
  scomp-link explain --artifact model.scomp --data test.csv
  scomp-link engineer --data raw.csv --target y --interactions --log-transform --output engineered.parquet
  scomp-link forecast --data series.csv --column value --horizon 30 --method auto
  scomp-link anomaly --data data.csv --methods iforest,lof --contamination 0.05
  scomp-link drift --reference train.csv --current production.csv
  scomp-link fairness --data preds.csv --target y_true --predicted y_pred --sensitive gender
  scomp-link quality --data raw_data.csv --output report.html
  scomp-link compare --artifacts v1.scomp v2.scomp
  scomp-link report --data train.csv --output eda_report.html
  scomp-link report --artifact model.scomp --data test.csv --output model_report.html
  scomp-link info --artifact model.scomp
""",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 1.1.3")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── run ──
    p_run = subparsers.add_parser("run", help="Run a complete ML pipeline",
                                   description="Train and evaluate a model on tabular data.")
    p_run.add_argument("--data", required=True, help="Path to input data (CSV, TSV, Parquet)")
    p_run.add_argument("--target", required=True, help="Target column name")
    p_run.add_argument("--task", required=True, choices=["regression", "classification"],
                       help="ML task type")
    p_run.add_argument("--features", default=None,
                       help="Comma-separated feature columns (default: all except target)")
    p_run.add_argument("--model-hint", default=None,
                       help="Model selection hint (e.g. numerical_prediction, categorical_known)")
    p_run.add_argument("--test-size", type=float, default=0.2,
                       help="Test split ratio (default: 0.2)")
    p_run.add_argument("--name", default=None, help="Pipeline name (default: filename)")
    p_run.add_argument("--output", "-o", default=None, help="Output path for results JSON")
    p_run.add_argument("--save-artifact", default=None,
                       help="Save trained pipeline as .scomp artifact")
    p_run.add_argument("--silent", action="store_true", help="Suppress progress output")
    p_run.add_argument("--engineer", action="store_true",
                       help="Apply automatic feature engineering before training")
    p_run.add_argument("--ensemble", choices=["voting", "stacking"], default=None,
                       help="Enable ensemble learning (voting or stacking)")
    p_run.add_argument("--advanced-cv", action="store_true",
                       help="Run advanced cross-validation (LOOCV + Bootstrap)")
    p_run.set_defaults(func=cmd_run)

    # ── predict ──
    p_pred = subparsers.add_parser("predict", help="Predict using a saved .scomp artifact",
                                    description="Load a pipeline artifact and generate predictions.")
    p_pred.add_argument("--artifact", required=True, help="Path to .scomp artifact")
    p_pred.add_argument("--data", required=True, help="Path to input data")
    p_pred.add_argument("--output", "-o", default=None, help="Output path (default: predictions.csv)")
    p_pred.add_argument("--silent", action="store_true", help="Suppress progress output")
    p_pred.set_defaults(func=cmd_predict)

    # ── explain ──
    p_exp = subparsers.add_parser("explain", help="Generate SHAP feature importance",
                                   description="Compute SHAP values for a trained model.")
    p_exp.add_argument("--artifact", required=True, help="Path to .scomp artifact")
    p_exp.add_argument("--data", required=True, help="Path to data for explanation")
    p_exp.add_argument("--n-samples", type=int, default=100,
                       help="Number of samples to explain (default: 100)")
    p_exp.add_argument("--output", "-o", default=None, help="Output path (default: feature_importance.csv)")
    p_exp.add_argument("--silent", action="store_true", help="Suppress progress output")
    p_exp.set_defaults(func=cmd_explain)

    # ── drift ──
    p_drift = subparsers.add_parser("drift", help="Detect data drift between datasets",
                                     description="Compare reference vs current data distributions.")
    p_drift.add_argument("--reference", required=True, help="Path to reference (training) data")
    p_drift.add_argument("--current", required=True, help="Path to current (production) data")
    p_drift.add_argument("--features", default=None,
                         help="Comma-separated features to check (default: all numeric)")
    p_drift.add_argument("--threshold", type=float, default=0.2,
                         help="PSI threshold for drift (default: 0.2)")
    p_drift.add_argument("--output", "-o", default=None, help="Output path for drift report CSV")
    p_drift.add_argument("--silent", action="store_true", help="Suppress progress output")
    p_drift.set_defaults(func=cmd_drift)

    # ── quality ──
    p_qual = subparsers.add_parser("quality", help="Generate data quality report",
                                    description="Profile a dataset and generate an HTML report.")
    p_qual.add_argument("--data", required=True, help="Path to data to profile")
    p_qual.add_argument("--output", "-o", default=None,
                        help="Output path (default: data_quality_report.html)")
    p_qual.add_argument("--silent", action="store_true", help="Suppress progress output")
    p_qual.set_defaults(func=cmd_quality)

    # ── info ──
    p_info = subparsers.add_parser("info", help="Inspect a .scomp artifact",
                                    description="Show metadata, config, and metrics of a saved pipeline.")
    p_info.add_argument("--artifact", required=True, help="Path to .scomp artifact")
    p_info.set_defaults(func=cmd_info)

    # ── report ──
    p_rep = subparsers.add_parser("report", help="Generate interactive HTML report",
                                   description="Generate an EDA report on a dataset, or a model evaluation report from an artifact.")
    p_rep.add_argument("--data", default=None, help="Path to data (required for EDA, optional for model report)")
    p_rep.add_argument("--artifact", default=None, help="Path to .scomp artifact (triggers model report mode)")
    p_rep.add_argument("--output", "-o", default=None, help="Output HTML path (default: report.html)")
    p_rep.add_argument("--silent", action="store_true", help="Suppress progress output")
    p_rep.set_defaults(func=cmd_report)

    # ── engineer ──
    p_eng = subparsers.add_parser("engineer", help="Apply feature engineering",
                                   description="Transform a dataset with automated feature engineering.")
    p_eng.add_argument("--data", required=True, help="Path to input data")
    p_eng.add_argument("--target", default=None, help="Target column (for target encoding)")
    p_eng.add_argument("--output", "-o", default=None, help="Output path (default: engineered.csv)")
    p_eng.add_argument("--interactions", action="store_true", help="Generate polynomial interactions")
    p_eng.add_argument("--log-transform", action="store_true", help="Log-transform skewed features")
    p_eng.add_argument("--date-features", action="store_true", help="Extract date components")
    p_eng.add_argument("--target-encode", action="store_true", help="Target-encode high-cardinality categoricals")
    p_eng.add_argument("--auto-bin", action="store_true", help="Auto-bin numeric features")
    p_eng.add_argument("--n-bins", type=int, default=5, help="Number of bins (default: 5)")
    p_eng.add_argument("--silent", action="store_true", help="Suppress progress output")
    p_eng.set_defaults(func=cmd_engineer)

    # ── forecast ──
    p_fc = subparsers.add_parser("forecast", help="Time series forecasting",
                                  description="Forecast future values from a time series column.")
    p_fc.add_argument("--data", required=True, help="Path to input data")
    p_fc.add_argument("--column", required=True, help="Column containing the time series values")
    p_fc.add_argument("--horizon", type=int, default=10, help="Forecast horizon (default: 10)")
    p_fc.add_argument("--method", default="auto", choices=["auto", "arima", "sarima", "exp_smoothing"],
                      help="Forecasting method (default: auto)")
    p_fc.add_argument("--seasonal-period", type=int, default=None, help="Seasonal period (default: auto-detect)")
    p_fc.add_argument("--cv-splits", type=int, default=None, help="Walk-forward CV splits (optional)")
    p_fc.add_argument("--output", "-o", default=None, help="Output path (default: forecast.csv)")
    p_fc.add_argument("--silent", action="store_true", help="Suppress progress output")
    p_fc.set_defaults(func=cmd_forecast)

    # ── anomaly ──
    p_anom = subparsers.add_parser("anomaly", help="Detect anomalies in tabular data",
                                    description="Multi-method consensus anomaly detection.")
    p_anom.add_argument("--data", required=True, help="Path to input data")
    p_anom.add_argument("--features", default=None, help="Comma-separated feature columns (default: all numeric)")
    p_anom.add_argument("--methods", default="iforest,lof",
                        help="Comma-separated methods: iforest,lof,tabnet,transformer (default: iforest,lof)")
    p_anom.add_argument("--contamination", type=float, default=0.05,
                        help="Expected anomaly fraction (default: 0.05)")
    p_anom.add_argument("--consensus", type=int, default=2,
                        help="Min methods that must agree (default: 2)")
    p_anom.add_argument("--output", "-o", default=None, help="Output path (default: anomalies.csv)")
    p_anom.add_argument("--silent", action="store_true", help="Suppress progress output")
    p_anom.set_defaults(func=cmd_anomaly)

    # ── fairness ──
    p_fair = subparsers.add_parser("fairness", help="Check fairness and bias metrics",
                                    description="Compute fairness metrics on model predictions.")
    p_fair.add_argument("--data", required=True, help="Path to data with predictions")
    p_fair.add_argument("--target", required=True, help="True label column")
    p_fair.add_argument("--predicted", required=True, help="Predicted label column")
    p_fair.add_argument("--sensitive", required=True, help="Protected attribute column")
    p_fair.add_argument("--output", "-o", default=None, help="Output path for report JSON")
    p_fair.add_argument("--silent", action="store_true", help="Suppress progress output")
    p_fair.set_defaults(func=cmd_fairness)

    # ── compare ──
    p_cmp = subparsers.add_parser("compare", help="Compare multiple model artifacts",
                                   description="Side-by-side comparison of .scomp artifacts.")
    p_cmp.add_argument("--artifacts", nargs="+", required=True, help="Paths to .scomp artifacts")
    p_cmp.add_argument("--output", "-o", default=None, help="Output path (default: stdout)")
    p_cmp.set_defaults(func=cmd_compare)

    # ── init ──
    p_init = subparsers.add_parser("init", help="Scaffold a new ML project",
                                    description="Generate a project directory with pipeline template, config, and folder structure.")
    p_init.add_argument("name", help="Project name (creates a directory with this name)")
    p_init.add_argument("--force", action="store_true", help="Overwrite existing directory")
    p_init.set_defaults(func=cmd_init)

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
