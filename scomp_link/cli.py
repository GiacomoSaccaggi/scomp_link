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
import json
import sys
from pathlib import Path
from typing import Optional


def _load_data(path: str, target: Optional[str] = None):
    """Load data from CSV/Parquet file."""
    import pandas as pd

    p = Path(path)
    if not p.exists():
        sys.exit(f"Error: file not found: {path}")
    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    elif p.suffix in (".csv", ".tsv"):
        sep = "\t" if p.suffix == ".tsv" else ","
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
        fmt = p.suffix.lstrip(".")

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
        with open(p, "w") as f:
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
        fe = FeatureEngineer(
            interactions=True, log_transform=True, date_features=True, target_encode=True, auto_bin=False
        )
        X_tmp = fe.fit_transform(X_tmp, y_tmp)
        X_tmp[args.target] = y_tmp.values
        df = X_tmp

    pipe = scomp_link.ScompLinkPipeline(args.name or Path(args.data).stem)
    pipe.set_objectives([f"Optimize {args.task}"])
    pipe.import_and_clean_data(df)

    feature_cols = args.features.split(",") if args.features else None
    pipe.select_variables(target_col=args.target, feature_cols=feature_cols)

    metadata = {}
    if args.model_hint:
        pipe.choose_model(args.model_hint, metadata=metadata)
    else:
        pipe.choose_model(
            "numerical_prediction" if args.task == "regression" else "categorical_known", metadata=metadata
        )

    kwargs = dict(task_type=args.task, test_size=args.test_size)
    if args.ensemble:
        kwargs["use_ensemble"] = True
        kwargs["ensemble_strategy"] = args.ensemble
    if args.advanced_cv:
        kwargs["advanced_cv"] = True
    if hasattr(args, "text_col") and args.text_col:
        kwargs["text_col"] = args.text_col
    if hasattr(args, "image_col") and args.image_col:
        kwargs["image_col"] = args.image_col
    if hasattr(args, "n_clusters") and args.n_clusters:
        kwargs["n_clusters"] = args.n_clusters

    results = pipe.run_pipeline(**kwargs)

    # Output
    output = {
        "status": results.get("status"),
        "model_type": results.get("model_type"),
        "metrics": results.get("metrics"),
        "report_path": results.get("report_path"),
    }

    fmt = getattr(args, "format", "json")
    if args.output:
        _format_output(output, fmt, args.output)
        print(f"Results saved to: {args.output}")
    else:
        _format_output(output, fmt)

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
    target_col = artifact.config.get("target_col")
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]

    predictions = artifact.predict(X)

    import pandas as pd

    out_df = df.copy()
    out_df["prediction"] = predictions

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
    target_col = artifact.config.get("target_col")
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]

    if artifact.preprocessor is not None:
        import pandas as pd

        X = pd.DataFrame(artifact.preprocessor.transform(X), columns=feature_cols)

    explainer = scomp_link.ShapExplainer(artifact.model, X[: min(100, len(X))])
    explainer.explain(X[: min(args.n_samples, len(X))])
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

    features = args.features.split(",") if args.features else None

    detector = scomp_link.DriftDetector(reference, psi_threshold=args.threshold)
    report = detector.detect(current, features=features)
    summary = detector.summary(report)

    if args.output:
        _write_output(report, args.output)
        print(f"Drift report saved to: {args.output}")
    else:
        print(report.to_string(index=False))

    print(
        f"\nSummary: {summary['drifted_features']}/{summary['total_features']} features drifted "
        f"(worst: {summary['worst_feature']}, PSI={summary['max_psi']:.4f})"
    )


def cmd_quality(args):
    """Generate a data quality report."""
    import scomp_link

    if args.silent:
        scomp_link.set_verbosity("silent")

    df = _load_data(args.data)
    dqr = scomp_link.DataQualityReport(df)
    report = dqr.generate()

    output_path = args.output or "data_quality_report.html"
    if output_path.endswith(".html"):
        dqr.save_html(output_path)
    else:
        _write_output(report["cardinality"], output_path)
    print(f"Report saved to: {output_path}")


def cmd_report(args):
    """Generate an interactive HTML report (EDA or model evaluation)."""
    import pandas as pd

    import scomp_link

    if args.silent:
        scomp_link.set_verbosity("silent")

    from scomp_link.utils.plotly_utils import barchart, histogram, linechart
    from scomp_link.utils.report_html import ScompLinkHTMLReport

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
        info_df = pd.DataFrame(
            [
                {
                    "model_type": type(artifact.model).__name__,
                    "task": artifact.config.get("task_type", "unknown"),
                    "target": artifact.config.get("target_col", "unknown"),
                    "n_features": len(artifact.feature_schema),
                }
            ]
        )
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
                explainer = scomp_link.ShapExplainer(artifact.model, X[: min(100, len(X))])
                explainer.explain(X[: min(50, len(X))])
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

                fig_scatter = px.scatter(
                    x=df[target_col],
                    y=predictions,
                    labels={"x": "Actual", "y": "Predicted"},
                    title="Predictions vs Actual",
                )
                fig_scatter.add_shape(
                    type="line",
                    x0=df[target_col].min(),
                    x1=df[target_col].max(),
                    y0=df[target_col].min(),
                    y1=df[target_col].max(),
                    line=dict(dash="dash", color="red"),
                )
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
        overview_df = pd.DataFrame(
            [
                {
                    "rows": len(df),
                    "columns": len(df.columns),
                    "numeric": len(df.select_dtypes(include=["number"]).columns),
                    "categorical": len(df.select_dtypes(include=["object", "category"]).columns),
                    "missing_total": int(df.isnull().sum().sum()),
                }
            ]
        )
        report.add_dataframe(overview_df, "Overview")
        report.close_section()

        # Distributions
        import plotly.express as px

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

        report.open_section("Feature Distributions")
        for col in numeric_cols[:12]:  # cap at 12
            fig = histogram(df[col].dropna().values, f"Distribution: {col}")
            report.add_graph_to_report(fig, col)
        report.close_section()

        # Correlations
        if len(numeric_cols) >= 2:
            report.open_section("Correlations")
            corr = df[numeric_cols].corr()
            fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", title="Correlation Matrix")
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
    import pandas as pd

    import scomp_link

    if args.silent:
        scomp_link.set_verbosity("silent")

    df = _load_data(args.data)
    if args.column not in df.columns:
        sys.exit(f"Error: column '{args.column}' not found. Available: {list(df.columns)}")

    series = df[args.column].dropna()
    fc = scomp_link.TimeSeriesForecaster(
        method=args.method,
        horizon=args.horizon,
        seasonal_period=args.seasonal_period,
    )

    if args.cv_splits:
        cv_results = fc.walk_forward_cv(series, n_splits=args.cv_splits, horizon=args.horizon)
        print(f"Walk-forward CV: MAE={cv_results['mean_mae']:.4f}, RMSE={cv_results['mean_rmse']:.4f}")

    fc.fit(series)
    ci = fc.predict_with_ci(steps=args.horizon)
    ci.index = range(len(series), len(series) + args.horizon)
    ci.index.name = "step"

    output_path = args.output or "forecast.csv"
    _write_output(ci.reset_index(), output_path)
    print(f"Forecast saved to: {output_path} ({args.horizon} steps)")

    if hasattr(args, "plot") and args.plot:
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=series.values, name="Historical"))
        fig.add_trace(
            go.Scatter(
                x=list(range(len(series), len(series) + args.horizon)),
                y=ci["forecast"].values,
                name="Forecast",
                line=dict(dash="dash"),
            )
        )
        if "lower" in ci.columns:
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(series), len(series) + args.horizon)),
                    y=ci["lower"].values,
                    fill=None,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(series), len(series) + args.horizon)),
                    y=ci["upper"].values,
                    fill="tonexty",
                    mode="lines",
                    line=dict(width=0),
                    name="95% CI",
                )
            )
        fig.update_layout(title="Time Series Forecast")
        fig.write_html(args.plot)
        print(f"Plot saved to: {args.plot}")


def cmd_anomaly(args):
    """Detect anomalies in tabular data."""
    import scomp_link

    if args.silent:
        scomp_link.set_verbosity("silent")

    df = _load_data(args.data)
    features = args.features.split(",") if args.features else df.select_dtypes(include=["number"]).columns.tolist()

    methods = args.methods.split(",") if args.methods else ["iforest", "lof"]
    detector = scomp_link.AnomalyDetector(
        contamination=args.contamination,
        methods=methods,
        consensus_threshold=args.consensus,
    )
    results = detector.fit_predict(df, features=features)

    output_path = args.output or "anomalies.csv"
    _write_output(results["data"], output_path)
    print(f"Anomaly detection saved to: {output_path}")
    print(results["comparison"].to_string(index=False))


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
        df[args.target].values, df[args.predicted].values, sensitive_feature=df[args.sensitive].values
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
    import pandas as pd

    import scomp_link

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

    if hasattr(args, "plot") and args.plot:
        import plotly.express as px

        numeric_cols = comparison.select_dtypes(include=["number"]).columns.tolist()
        if numeric_cols:
            fig = px.bar(comparison, x="artifact", y=numeric_cols, barmode="group", title="Model Comparison")
            fig.write_html(args.plot)
            print(f"Comparison plot saved to: {args.plot}")


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
    (project_dir / "config.yaml").write_text(f"""# {args.name} configuration
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
""")

    # .gitignore
    (project_dir / ".gitignore").write_text("""__pycache__/
*.py[cod]
.venv/
*.scomp
*.html
data/*.csv
data/*.parquet
models/
reports/
""")

    # README
    (project_dir / "README.md").write_text(f"""# {args.name}

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
""")

    print(f"✅ Project '{args.name}' created with structure:")
    print(f"   {args.name}/")
    print("   ├── data/")
    print("   ├── models/")
    print("   ├── reports/")
    print("   ├── pipeline.py")
    print("   ├── config.yaml")
    print("   ├── .gitignore")
    print("   └── README.md")
    print("\n   Next: add data to data/, edit pipeline.py, and run it.")


def cmd_text(args):
    """Run text classification on a dataset."""
    import scomp_link

    if args.silent:
        scomp_link.set_verbosity("silent")

    df = _load_data(args.data, args.target)
    if args.text_col not in df.columns:
        sys.exit(f"Error: text column '{args.text_col}' not found. Available: {list(df.columns)}")

    pipe = scomp_link.ScompLinkPipeline(args.name or Path(args.data).stem)
    pipe.set_objectives(["Text Classification"])
    pipe.import_and_clean_data(df)
    pipe.select_variables(target_col=args.target)
    pipe.choose_model("categorical_known")

    use_contrastive = args.method == "contrastive"
    results = pipe.run_pipeline(
        task_type="text",
        text_col=args.text_col,
        use_contrastive=use_contrastive,
        epochs=args.epochs,
        batch_size=args.batch_size,
        test_size=args.test_size,
        text_model=args.model_name,
    )

    output = {
        "status": results.get("status"),
        "model_type": results.get("model_type"),
        "metrics": results.get("metrics"),
    }

    if args.output:
        _write_output(output, args.output, fmt="json")
        print(f"Results saved to: {args.output}")
    else:
        print(json.dumps(output, indent=2, default=str))

    if args.save_artifact:
        artifact = scomp_link.ScompArtifact()
        artifact.set_model(pipe.model)
        artifact.set_config(task_type="text", target_col=args.target, text_col=args.text_col)
        artifact.set_metrics(results.get("metrics", {}))
        artifact.save(args.save_artifact)
        print(f"Artifact saved to: {args.save_artifact}")


def cmd_cluster(args):
    """Run clustering on a dataset."""
    import pandas as pd

    import scomp_link

    if args.silent:
        scomp_link.set_verbosity("silent")

    df = _load_data(args.data)
    features = args.features.split(",") if args.features else df.select_dtypes(include=["number"]).columns.tolist()

    pipe = scomp_link.ScompLinkPipeline(args.name or Path(args.data).stem)
    pipe.set_objectives(["Clustering"])
    pipe.import_and_clean_data(df[features].copy().assign(_dummy=0))
    pipe.select_variables(target_col="_dummy", feature_cols=features)

    if args.method == "meanshift":
        pipe.choose_model("categorical_unknown", metadata={"categories_known": False})
    else:
        pipe.choose_model("categorical_unknown", metadata={"categories_known": True})

    results = pipe.run_pipeline(task_type="clustering", n_clusters=args.n_clusters)

    # Results come from cleaned data (may have fewer rows due to outlier removal)
    out_df = pd.DataFrame({"cluster": results["clusters"]})
    # If lengths match, merge with original features
    if len(results["clusters"]) == len(df):
        out_df = df.copy()
        out_df["cluster"] = results["clusters"]

    output_path = args.output or "clusters.csv"
    _write_output(out_df, output_path)
    print(f"Clusters saved to: {output_path}")
    print(f"Number of clusters: {results['n_clusters']}")
    print(f"Silhouette score: {results['metrics'].get('silhouette_score', 'N/A'):.4f}")

    if args.plot:
        import plotly.express as px

        if len(features) >= 2 and features[0] in out_df.columns:
            out_df["cluster"] = out_df["cluster"].astype(str)
            fig = px.scatter(out_df, x=features[0], y=features[1], color="cluster", title="Cluster Visualization")
            fig.write_html(args.plot)
            print(f"Plot saved to: {args.plot}")
        elif len(features) >= 2:
            # Rebuild plot df from cleaned data
            plot_df = pd.DataFrame({"cluster": results["clusters"]})
            plot_df["cluster"] = plot_df["cluster"].astype(str)
            # Use first two features from the pipeline's cleaned data
            pipe_df = pipe.df[features[:2]].reset_index(drop=True)
            plot_df = pd.concat([pipe_df, plot_df], axis=1)
            fig = px.scatter(plot_df, x=features[0], y=features[1], color="cluster", title="Cluster Visualization")
            fig.write_html(args.plot)
            print(f"Plot saved to: {args.plot}")


def cmd_tune(args):
    """Run hyperparameter tuning."""
    import scomp_link

    if args.silent:
        scomp_link.set_verbosity("silent")

    df = _load_data(args.data, args.target)
    feature_cols = args.features.split(",") if args.features else [c for c in df.columns if c != args.target]

    X = df[feature_cols]
    y = df[args.target]

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42)

    if args.method == "optuna":
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

        from scomp_link.models.advanced_tuning import OptunaOptimizer

        if args.task == "regression":
            base_model = GradientBoostingRegressor
            scoring = "r2"
        else:
            base_model = GradientBoostingClassifier
            scoring = "accuracy"

        def param_space(trial):
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            }

        optimizer = OptunaOptimizer(base_model, param_space, scoring=scoring, n_trials=args.n_trials)
        best_model = optimizer.optimize(X_train, y_train)

    elif args.method == "halving":
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

        from scomp_link.models.advanced_tuning import HalvingSearchOptimizer

        if args.task == "regression":
            model = GradientBoostingRegressor(random_state=42)
            scoring = "r2"
        else:
            model = GradientBoostingClassifier(random_state=42)
            scoring = "accuracy"

        param_grid = {
            "n_estimators": [50, 100, 200],
            "max_depth": [3, 5, 8],
            "learning_rate": [0.01, 0.05, 0.1],
        }

        halving_opt = HalvingSearchOptimizer(model, param_grid, scoring=scoring)
        best_model = halving_opt.optimize(X_train, y_train)
    else:
        sys.exit(f"Error: unsupported tuning method '{args.method}'. Use: optuna, halving")

    # Evaluate best model
    y_pred = best_model.predict(X_test)
    if args.task == "regression":
        from sklearn.metrics import mean_squared_error, r2_score

        metrics = {"r2": float(r2_score(y_test, y_pred)), "rmse": float(mean_squared_error(y_test, y_pred) ** 0.5)}
    else:
        from sklearn.metrics import accuracy_score, f1_score

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        }

    fmt = getattr(args, "format", "json")
    tune_output = {"method": args.method, "metrics": metrics}
    if args.output:
        _format_output(tune_output, fmt, args.output)
        print(f"Results saved to: {args.output}")
    else:
        _format_output(tune_output, fmt)

    if args.save_artifact:
        artifact = scomp_link.ScompArtifact()
        artifact.set_model(best_model)
        artifact.set_config(task_type=args.task, target_col=args.target)
        artifact.set_metrics(metrics)
        artifact.set_feature_schema(X_train)
        artifact.save(args.save_artifact)
        print(f"Best model saved to: {args.save_artifact}")


def cmd_validate(args):
    """Validate a model artifact on test data."""
    import scomp_link

    if args.silent:
        scomp_link.set_verbosity("silent")

    if not Path(args.artifact).exists():
        sys.exit(f"Error: artifact not found: {args.artifact}")

    scomp_link.set_verbosity("silent")
    artifact = scomp_link.ScompArtifact.load(args.artifact)
    if not args.silent:
        scomp_link.set_verbosity("info")

    df = _load_data(args.data, args.target)
    feature_cols = [c for c in df.columns if c != args.target]
    X = df[feature_cols]
    y = df[args.target]

    predictions = artifact.predict(X)

    task_type = artifact.config.get("task_type", "regression")
    validator = scomp_link.Validator(artifact.model)
    metrics = validator.evaluate(y, predictions, task_type=task_type)

    output = {"task_type": task_type, "metrics": metrics, "n_samples": len(y)}

    fmt = getattr(args, "format", "json")
    if args.output:
        _format_output(output, fmt, args.output)
        print(f"Metrics saved to: {args.output}")
    else:
        _format_output(output, fmt)

    if args.report:
        y_proba = None
        if task_type == "classification" and hasattr(artifact.model, "predict_proba"):
            y_proba = artifact.model.predict_proba(X)
        validator.generate_validation_report(
            y, predictions, task_type=task_type, y_proba=y_proba, report_name=args.report
        )
        print(f"Report saved to: {args.report}")


def cmd_monitor(args):
    """Run production monitoring: drift + quality + optional performance."""
    import pandas as pd

    import scomp_link

    if args.silent:
        scomp_link.set_verbosity("silent")

    if not Path(args.reference).exists():
        sys.exit(f"Error: reference data not found: {args.reference}")
    if not Path(args.current).exists():
        sys.exit(f"Error: current data not found: {args.current}")

    df_ref = _load_data(args.reference)
    df_cur = _load_data(args.current)

    from scomp_link.utils.plotly_utils import barchart
    from scomp_link.utils.report_html import ScompLinkHTMLReport

    report = ScompLinkHTMLReport(title="Production Monitoring Report")

    # 1. Data Quality on current data
    report.open_section("Data Quality - Current Batch")
    dqr = scomp_link.DataQualityReport(df_cur)
    quality = dqr.generate()
    overview_df = pd.DataFrame([quality["overview"]])
    report.add_dataframe(overview_df, "Quality Overview")
    if len(quality["missing"]) > 0:
        report.add_dataframe(quality["missing"], "Missing Values")
    report.close_section()

    # 2. Drift Detection
    report.open_section("Distribution Drift")
    numeric_cols = df_ref.select_dtypes(include=["number"]).columns.tolist()
    common_cols = [c for c in numeric_cols if c in df_cur.columns]
    if common_cols:
        detector = scomp_link.DriftDetector(df_ref[common_cols], psi_threshold=args.threshold)
        drift_report = detector.detect(df_cur[common_cols])
        report.add_dataframe(drift_report, "Drift Report")
        summary = detector.summary(drift_report)
        fig = detector.plot_drift_report(drift_report)
        report.add_graph_to_report(fig, "PSI by Feature")
        report.add_text(f"Drifted features: {summary['drifted_features']}/{summary['total_features']}")
    else:
        report.add_text("No common numeric columns found for drift detection.")
    report.close_section()

    # 3. Model Performance (if artifact + target provided)
    if args.artifact and args.target:
        artifact_path = Path(args.artifact)
        if artifact_path.exists() and args.target in df_cur.columns:
            report.open_section("Model Performance")
            scomp_link.set_verbosity("silent")
            artifact = scomp_link.ScompArtifact.load(args.artifact)
            if not args.silent:
                scomp_link.set_verbosity("info")

            feature_cols = [c for c in df_cur.columns if c != args.target]
            X_cur = df_cur[feature_cols]
            y_cur = df_cur[args.target]
            predictions = artifact.predict(X_cur)

            task_type = artifact.config.get("task_type", "regression")
            validator = scomp_link.Validator(artifact.model)
            metrics = validator.evaluate(y_cur, predictions, task_type=task_type)

            metrics_df = pd.DataFrame([metrics])
            report.add_dataframe(metrics_df, "Current Batch Metrics")
            report.close_section()

    output_path = args.output or "monitor_report.html"
    report.save_html(output_path)
    print(f"Monitoring report saved to: {output_path}")


def cmd_list_models(args):
    """List all available model types."""
    models = [
        (
            "Regression",
            [
                "Econometric Model (LinearRegression) - <1k records",
                "Ridge / SVR - numerical features, all important",
                "Lasso / Elastic Net - numerical features, feature selection",
                "Gradient Boosting / Random Forest - mixed features",
                "SGD Regressor - >100k records, numerical",
            ],
        ),
        (
            "Classification",
            [
                "Naive Bayes / Classification Tree - categorical, <5 features",
                "SVC / K-Neighbors / Naive Bayes - mixed, <300 per category",
                "SGD / Gradient Boosting / Random Forest - mixed, >300 per category",
                "Pre-trained model - images, <500 per category",
                "CNN (ResNet/Inception) - images, >500 per category",
            ],
        ),
        (
            "Clustering",
            [
                "KMeans / Hierarchical Clustering - known n_clusters",
                "Mean-Shift Clustering - unknown n_clusters",
            ],
        ),
        (
            "Text",
            [
                "TF-IDF + SGD Classifier - fast, lightweight",
                "Contrastive Text Classifier (BERT) - high accuracy, heavy",
            ],
        ),
        (
            "Time Series",
            [
                "UCM State Space - numerical study",
                "VAR / VARMA - multi-target prediction",
                "ARIMA / Exp. Smoothing - forecasting",
            ],
        ),
        (
            "Anomaly Detection",
            [
                "Isolation Forest - tree-based",
                "Local Outlier Factor - density-based",
                "TabNet Autoencoder - neural attention",
                "Transformer Autoencoder - self-attention",
            ],
        ),
    ]
    print("\nAvailable Models in scomp-link:\n")
    for category, model_list in models:
        print(f"  -- {category} --")
        for m in model_list:
            print(f"    * {m}")
        print()


def cmd_check_deps(args):
    """Check which optional dependencies are installed."""
    deps = [
        ("numpy", "Core"),
        ("pandas", "Core"),
        ("scikit-learn", "Core"),
        ("plotly", "Core"),
        ("torch", "NLP / Deep Learning"),
        ("transformers", "NLP (BERT)"),
        ("spacy", "NLP (SpacyEmbeddingModel)"),
        ("faiss", "NLP (FAISS similarity)"),
        ("sentence_transformers", "NLP (Embeddings)"),
        ("tensorflow", "Computer Vision (CNN)"),
        ("pytorch_tabnet", "Anomaly Detection (TabNet)"),
        ("statsmodels", "Time Series"),
        ("optuna", "Hyperparameter Tuning"),
        ("shap", "Explainability (SHAP)"),
        ("lime", "Explainability (LIME)"),
        ("weasyprint", "PDF Export"),
        ("playwright", "PDF Export (Playwright)"),
        ("polars", "Preprocessing (fast)"),
    ]
    print("\nDependency Check:\n")
    installed = 0
    for pkg, purpose in deps:
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "?")
            print(f"  OK  {pkg:<22} {version:<12} ({purpose})")
            installed += 1
        except (ImportError, OSError):
            print(f"  --  {pkg:<22} {'missing':<12} ({purpose})")
    print(f"\n  {installed}/{len(deps)} packages available.")


def _format_output(data, fmt: str, output_path: str = None):
    """Format and print/save output in json, csv, or table format."""
    import pandas as pd

    if fmt == "json":
        text = json.dumps(data, indent=2, default=str)
        if output_path:
            Path(output_path).write_text(text)
        else:
            print(text)
    elif fmt == "csv":
        if isinstance(data, pd.DataFrame):
            df = data
        elif isinstance(data, dict):
            df = (
                pd.DataFrame([data])
                if not any(isinstance(v, (list, dict)) for v in data.values())
                else pd.json_normalize(data)
            )
        else:
            df = pd.DataFrame(data)
        if output_path:
            df.to_csv(output_path, index=False)
        else:
            print(df.to_csv(index=False))
    elif fmt == "table":
        if isinstance(data, pd.DataFrame):
            df = data
        elif isinstance(data, dict):
            df = (
                pd.DataFrame([data])
                if not any(isinstance(v, (list, dict)) for v in data.values())
                else pd.json_normalize(data)
            )
        else:
            df = pd.DataFrame(data)
        if output_path:
            Path(output_path).write_text(df.to_string(index=False))
        else:
            print(df.to_string(index=False))
    else:
        sys.exit(f"Error: unsupported format '{fmt}'. Use: json, csv, table")


def cmd_serve(args):
    """Serve a .scomp artifact as a REST API."""
    import scomp_link

    if not Path(args.artifact).exists():
        sys.exit(f"Error: artifact not found: {args.artifact}")

    try:
        from flask import Flask, jsonify, request
    except ImportError:
        sys.exit("Error: flask required for serving. Install with: pip install flask")

    scomp_link.set_verbosity("silent")
    artifact = scomp_link.ScompArtifact.load(args.artifact)

    app = Flask("scomp-link-serve")

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "model": type(artifact.model).__name__})

    @app.route("/info", methods=["GET"])
    def info():
        return jsonify(artifact.info())

    @app.route("/predict", methods=["POST"])
    def predict():
        import numpy as np
        import pandas as pd

        data = request.get_json(force=True)
        if "instances" in data:
            df = pd.DataFrame(data["instances"])
        elif "data" in data:
            df = pd.DataFrame(data["data"])
        else:
            df = pd.DataFrame([data])

        predictions = artifact.predict(df)
        if hasattr(predictions, "tolist"):
            predictions = predictions.tolist()
        return jsonify({"predictions": predictions})

    @app.route("/schema", methods=["GET"])
    def schema():
        return jsonify({"features": artifact.feature_schema, "config": artifact.config})

    print(f"Serving model: {type(artifact.model).__name__}")
    print(f"Config: task={artifact.config.get('task_type')}, target={artifact.config.get('target_col')}")
    print("Endpoints:")
    print(f"  GET  http://{args.host}:{args.port}/health")
    print(f"  GET  http://{args.host}:{args.port}/info")
    print(f"  GET  http://{args.host}:{args.port}/schema")
    print(f"  POST http://{args.host}:{args.port}/predict")
    print(f"\nStarting server on {args.host}:{args.port}...")
    app.run(host=args.host, port=args.port, debug=args.debug)


def cmd_pipeline(args):
    """Run a multi-step ML pipeline from a YAML config file."""
    import pandas as pd

    import scomp_link

    config_path = Path(args.config)
    if not config_path.exists():
        sys.exit(f"Error: config file not found: {args.config}")

    try:
        import yaml
    except ImportError:
        # Fallback: use simple YAML-like parsing or require install
        sys.exit("Error: PyYAML required for pipeline configs. Install with: pip install pyyaml")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Validate required fields
    for field in ["data", "target", "task"]:
        if field not in cfg:
            sys.exit(f"Error: required field '{field}' missing in config file")

    if args.silent:
        scomp_link.set_verbosity("silent")

    print(f"Running pipeline from: {args.config}")
    print(f"  Task: {cfg['task']}, Target: {cfg['target']}")

    # Step 1: Load data
    df = _load_data(cfg["data"], cfg["target"])
    print(f"  Data loaded: {df.shape}")

    # Step 2: Feature engineering (optional)
    if "engineer" in cfg:
        eng_cfg = cfg["engineer"]
        fe = scomp_link.FeatureEngineer(
            interactions=eng_cfg.get("interactions", False),
            log_transform=eng_cfg.get("log_transform", False),
            date_features=eng_cfg.get("date_features", False),
            target_encode=eng_cfg.get("target_encode", False),
            auto_bin=eng_cfg.get("auto_bin", False),
        )
        y = df[cfg["target"]]
        X = df.drop(columns=[cfg["target"]])
        X = fe.fit_transform(X, y)
        df = X.copy()
        df[cfg["target"]] = y.values
        print(f"  Feature engineering applied: {df.shape}")

    # Step 3: Tune (optional)
    if "tune" in cfg:
        tune_cfg = cfg["tune"]
        method = tune_cfg.get("method", "optuna")
        n_trials = tune_cfg.get("n_trials", 50)
        feature_cols = [c for c in df.columns if c != cfg["target"]]
        X = df[feature_cols]
        y = df[cfg["target"]]

        from sklearn.model_selection import train_test_split

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if method == "optuna":
            from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

            from scomp_link.models.advanced_tuning import OptunaOptimizer

            base = GradientBoostingRegressor if cfg["task"] == "regression" else GradientBoostingClassifier
            scoring = "r2" if cfg["task"] == "regression" else "accuracy"

            def param_space(trial):
                return {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 3, 12),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                }

            optimizer = OptunaOptimizer(base, param_space, scoring=scoring, n_trials=n_trials)
            best_model = optimizer.optimize(X_train, y_train)
        else:
            from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

            from scomp_link.models.advanced_tuning import HalvingSearchOptimizer

            model = (
                GradientBoostingRegressor(random_state=42)
                if cfg["task"] == "regression"
                else GradientBoostingClassifier(random_state=42)
            )
            scoring = "r2" if cfg["task"] == "regression" else "accuracy"
            param_grid = {"n_estimators": [50, 100, 200], "max_depth": [3, 5, 8], "learning_rate": [0.01, 0.05, 0.1]}
            halving_opt = HalvingSearchOptimizer(model, param_grid, scoring=scoring)
            best_model = halving_opt.optimize(X_train, y_train)

        y_pred = best_model.predict(X_test)
        if cfg["task"] == "regression":
            from sklearn.metrics import r2_score

            print(f"  Tuning complete ({method}, {n_trials} trials): R2={r2_score(y_test, y_pred):.4f}")
        else:
            from sklearn.metrics import accuracy_score

            print(f"  Tuning complete ({method}, {n_trials} trials): Accuracy={accuracy_score(y_test, y_pred):.4f}")

    elif "tune" not in cfg:
        # Standard pipeline without tuning
        pipe = scomp_link.ScompLinkPipeline(cfg.get("name", Path(cfg["data"]).stem))
        pipe.set_objectives([f"Optimize {cfg['task']}"])
        pipe.import_and_clean_data(df)
        pipe.select_variables(target_col=cfg["target"])

        model_hint = cfg.get(
            "model_hint", "numerical_prediction" if cfg["task"] == "regression" else "categorical_known"
        )
        pipe.choose_model(model_hint)

        kwargs = dict(task_type=cfg["task"], test_size=cfg.get("test_size", 0.2))
        if cfg.get("ensemble"):
            kwargs["use_ensemble"] = True
            kwargs["ensemble_strategy"] = cfg["ensemble"]
        if cfg.get("advanced_cv"):
            kwargs["advanced_cv"] = True

        results = pipe.run_pipeline(**kwargs)
        best_model = pipe.model
        print(f"  Pipeline complete: {results.get('metrics', {})}")

    # Step 4: Validate (optional)
    if "validate" in cfg:
        val_cfg = cfg["validate"]
        if "test_data" in val_cfg:
            test_df = _load_data(val_cfg["test_data"], cfg["target"])
            X_val = test_df.drop(columns=[cfg["target"]])
            y_val = test_df[cfg["target"]]
            preds = best_model.predict(X_val)
            validator = scomp_link.Validator(best_model)
            metrics = validator.evaluate(y_val, preds, task_type=cfg["task"])
            print(f"  Validation: {metrics}")
            if "report" in val_cfg:
                validator.generate_validation_report(y_val, preds, task_type=cfg["task"], report_name=val_cfg["report"])
                print(f"  Report saved: {val_cfg['report']}")

    # Step 5: Save (optional)
    if "save" in cfg:
        artifact = scomp_link.ScompArtifact()
        artifact.set_model(best_model)
        artifact.set_config(task_type=cfg["task"], target_col=cfg["target"])
        artifact.save(cfg["save"])
        print(f"  Artifact saved: {cfg['save']}")

    print("Pipeline complete.")


def cmd_describe(args):
    """Quick dataset profiling — one row per column."""
    import pandas as pd

    df = _load_data(args.data)

    rows = []
    for col in df.columns:
        row = {
            "column": col,
            "dtype": str(df[col].dtype),
            "missing%": round(df[col].isnull().mean() * 100, 1),
            "unique": df[col].nunique(),
        }
        if pd.api.types.is_numeric_dtype(df[col]):
            row["min"] = round(df[col].min(), 4)
            row["max"] = round(df[col].max(), 4)
            row["mean"] = round(df[col].mean(), 4)
            row["std"] = round(df[col].std(), 4)
        else:
            row["min"] = ""
            row["max"] = ""
            row["mean"] = ""
            row["std"] = ""
            # Show top value for categoricals
            if df[col].nunique() < 50:
                row["min"] = f"top: {df[col].mode().iloc[0]}" if len(df[col].mode()) > 0 else ""

        rows.append(row)

    result = pd.DataFrame(rows)

    fmt = getattr(args, "format", "table")
    if args.output:
        _write_output(result, args.output)
        print(f"Description saved to: {args.output}")
    else:
        _format_output(result, fmt)


def cmd_export(args):
    """Export a model to standard formats (pickle, ONNX, PMML)."""
    import scomp_link

    if not Path(args.artifact).exists():
        sys.exit(f"Error: artifact not found: {args.artifact}")

    scomp_link.set_verbosity("silent")
    artifact = scomp_link.ScompArtifact.load(args.artifact)
    model = artifact.model

    output_path = Path(args.output) if args.output else Path(args.artifact).with_suffix(f".{args.format}")

    if args.format == "pickle":
        import pickle

        with open(output_path, "wb") as f:
            pickle.dump(model, f)
        print(f"Model exported as pickle: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")

    elif args.format == "onnx":
        try:
            import skl2onnx
            from skl2onnx import convert_sklearn
            from skl2onnx.common.data_types import FloatTensorType
        except ImportError:
            sys.exit("Error: skl2onnx required for ONNX export. Install with: pip install skl2onnx")

        n_features = len(artifact.feature_schema)
        initial_type = [("X", FloatTensorType([None, n_features]))]
        try:
            onnx_model = convert_sklearn(model, initial_types=initial_type)
            with open(output_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            print(f"Model exported as ONNX: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")
        except Exception as e:
            sys.exit(f"Error: ONNX conversion failed: {e}")

    elif args.format == "pmml":
        try:
            from sklearn2pmml import PMMLPipeline, sklearn2pmml
        except ImportError:
            sys.exit("Error: sklearn2pmml required for PMML export. Install with: pip install sklearn2pmml")

        try:
            pmml_pipeline = PMMLPipeline([("model", model)])
            sklearn2pmml(pmml_pipeline, str(output_path), with_repr=True)
            print(f"Model exported as PMML: {output_path}")
        except Exception as e:
            sys.exit(f"Error: PMML conversion failed: {e}")

    elif args.format == "joblib":
        import joblib

        joblib.dump(model, output_path)
        print(f"Model exported as joblib: {output_path} ({output_path.stat().st_size / 1024:.1f} KB)")

    else:
        sys.exit(f"Error: unsupported export format '{args.format}'. Use: pickle, onnx, pmml, joblib")


def cmd_mcp(args):
    """Start the MCP (Model Context Protocol) server for agent integration."""
    try:
        from scomp_link.mcp_server import main as mcp_main

        mcp_main()
    except ImportError:
        sys.exit("Error: MCP dependencies not installed. Install with: pip install scomp-link[mcp]")


# ═══════════════════════════════════════════════════════════════════
# PARSER
# ═══════════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="scomp-link",
        description="scomp-link: End-to-end ML toolkit CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands (25 total):

  Training & Prediction:
    run             Train a model (regression, classification, text, clustering, image)
    predict         Generate predictions from a saved .scomp artifact
    text            Dedicated text classification (TF-IDF or BERT contrastive)
    cluster         KMeans or MeanShift clustering with optional scatter plot
    tune            Hyperparameter tuning (Optuna Bayesian or HalvingGridSearch)

  Evaluation & Monitoring:
    validate        Evaluate a saved artifact on test data with metrics + report
    explain         SHAP feature importance for a trained model
    fairness        Bias and fairness metrics (demographic parity, disparate impact)
    monitor         Combined drift + quality + performance monitoring report
    compare         Side-by-side comparison of multiple artifacts

  Data & Features:
    quality         Full data quality HTML report (missing, duplicates, correlations)
    describe        Quick column-level profiling (dtype, missing%%, unique, stats)
    engineer        Automated feature engineering (interactions, log, dates, encoding)
    drift           Detect distribution shift between reference and current data
    anomaly         Multi-method consensus anomaly detection
    forecast        Time series forecasting (ARIMA, Exponential Smoothing, auto)

  Model Lifecycle:
    init            Scaffold a new ML project with template files
    serve           Deploy a .scomp artifact as REST API (Flask)
    export          Convert .scomp to pickle, ONNX, PMML, or joblib
    report          Generate interactive HTML report (EDA or model evaluation)
    pipeline        Run a full pipeline from a YAML config file
    info            Inspect a .scomp artifact metadata

  Utilities:
    list-models     Show all available model types
    check-deps      Check which optional dependencies are installed

Examples:
  scomp-link run --data train.csv --target price --task regression --format table
  scomp-link run --data data.csv --target label --task text --text-col message
  scomp-link tune --data train.csv --target y --task regression --method optuna --n-trials 100
  scomp-link validate --artifact model.scomp --data test.csv --target y --report report.html
  scomp-link serve --artifact model.scomp --port 8080
  scomp-link pipeline --config pipeline.yaml
  scomp-link describe --data dataset.csv --format csv
  scomp-link export --artifact model.scomp --format onnx
  scomp-link cluster --data customers.csv --n-clusters 5 --plot clusters.html
  scomp-link monitor --reference train.csv --current prod.csv --artifact model.scomp --target y
  scomp-link forecast --data series.csv --column value --horizon 30 --plot forecast.html
""",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 1.2.5")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ── run ──
    p_run = subparsers.add_parser(
        "run", help="Run a complete ML pipeline", description="Train and evaluate a model on tabular data."
    )
    p_run.add_argument("--data", required=True, help="Path to input data (CSV, TSV, Parquet)")
    p_run.add_argument("--target", required=True, help="Target column name")
    p_run.add_argument(
        "--task",
        required=True,
        choices=["regression", "classification", "text", "clustering", "image"],
        help="ML task type",
    )
    p_run.add_argument("--features", default=None, help="Comma-separated feature columns (default: all except target)")
    p_run.add_argument("--text-col", default=None, help="Text column (for --task text)")
    p_run.add_argument("--image-col", default=None, help="Image column (for --task image)")
    p_run.add_argument("--n-clusters", type=int, default=None, help="Number of clusters (for --task clustering)")
    p_run.add_argument(
        "--model-hint", default=None, help="Model selection hint (e.g. numerical_prediction, categorical_known)"
    )
    p_run.add_argument("--test-size", type=float, default=0.2, help="Test split ratio (default: 0.2)")
    p_run.add_argument("--name", default=None, help="Pipeline name (default: filename)")
    p_run.add_argument("--output", "-o", default=None, help="Output path for results JSON")
    p_run.add_argument("--save-artifact", default=None, help="Save trained pipeline as .scomp artifact")
    p_run.add_argument(
        "--format", choices=["json", "csv", "table"], default="json", help="Output format for results (default: json)"
    )
    p_run.add_argument("--silent", action="store_true", help="Suppress progress output")
    p_run.add_argument("--engineer", action="store_true", help="Apply automatic feature engineering before training")
    p_run.add_argument(
        "--ensemble", choices=["voting", "stacking"], default=None, help="Enable ensemble learning (voting or stacking)"
    )
    p_run.add_argument("--advanced-cv", action="store_true", help="Run advanced cross-validation (LOOCV + Bootstrap)")
    p_run.set_defaults(func=cmd_run)

    # ── predict ──
    p_pred = subparsers.add_parser(
        "predict",
        help="Predict using a saved .scomp artifact",
        description="Load a pipeline artifact and generate predictions.",
    )
    p_pred.add_argument("--artifact", required=True, help="Path to .scomp artifact")
    p_pred.add_argument("--data", required=True, help="Path to input data")
    p_pred.add_argument("--output", "-o", default=None, help="Output path (default: predictions.csv)")
    p_pred.add_argument("--silent", action="store_true", help="Suppress progress output")
    p_pred.set_defaults(func=cmd_predict)

    # ── explain ──
    p_exp = subparsers.add_parser(
        "explain", help="Generate SHAP feature importance", description="Compute SHAP values for a trained model."
    )
    p_exp.add_argument("--artifact", required=True, help="Path to .scomp artifact")
    p_exp.add_argument("--data", required=True, help="Path to data for explanation")
    p_exp.add_argument("--n-samples", type=int, default=100, help="Number of samples to explain (default: 100)")
    p_exp.add_argument("--output", "-o", default=None, help="Output path (default: feature_importance.csv)")
    p_exp.add_argument("--silent", action="store_true", help="Suppress progress output")
    p_exp.set_defaults(func=cmd_explain)

    # ── drift ──
    p_drift = subparsers.add_parser(
        "drift",
        help="Detect data drift between datasets",
        description="Compare reference vs current data distributions.",
    )
    p_drift.add_argument("--reference", required=True, help="Path to reference (training) data")
    p_drift.add_argument("--current", required=True, help="Path to current (production) data")
    p_drift.add_argument("--features", default=None, help="Comma-separated features to check (default: all numeric)")
    p_drift.add_argument("--threshold", type=float, default=0.2, help="PSI threshold for drift (default: 0.2)")
    p_drift.add_argument("--output", "-o", default=None, help="Output path for drift report CSV")
    p_drift.add_argument("--plot", default=None, help="Save drift plot as HTML")
    p_drift.add_argument("--silent", action="store_true", help="Suppress progress output")
    p_drift.set_defaults(func=cmd_drift)

    # ── quality ──
    p_qual = subparsers.add_parser(
        "quality", help="Generate data quality report", description="Profile a dataset and generate an HTML report."
    )
    p_qual.add_argument("--data", required=True, help="Path to data to profile")
    p_qual.add_argument("--output", "-o", default=None, help="Output path (default: data_quality_report.html)")
    p_qual.add_argument("--silent", action="store_true", help="Suppress progress output")
    p_qual.set_defaults(func=cmd_quality)

    # ── info ──
    p_info = subparsers.add_parser(
        "info", help="Inspect a .scomp artifact", description="Show metadata, config, and metrics of a saved pipeline."
    )
    p_info.add_argument("--artifact", required=True, help="Path to .scomp artifact")
    p_info.set_defaults(func=cmd_info)

    # ── report ──
    p_rep = subparsers.add_parser(
        "report",
        help="Generate interactive HTML report",
        description="Generate an EDA report on a dataset, or a model evaluation report from an artifact.",
    )
    p_rep.add_argument("--data", default=None, help="Path to data (required for EDA, optional for model report)")
    p_rep.add_argument("--artifact", default=None, help="Path to .scomp artifact (triggers model report mode)")
    p_rep.add_argument("--output", "-o", default=None, help="Output HTML path (default: report.html)")
    p_rep.add_argument("--silent", action="store_true", help="Suppress progress output")
    p_rep.set_defaults(func=cmd_report)

    # ── engineer ──
    p_eng = subparsers.add_parser(
        "engineer",
        help="Apply feature engineering",
        description="Transform a dataset with automated feature engineering.",
    )
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
    p_fc = subparsers.add_parser(
        "forecast", help="Time series forecasting", description="Forecast future values from a time series column."
    )
    p_fc.add_argument("--data", required=True, help="Path to input data")
    p_fc.add_argument("--column", required=True, help="Column containing the time series values")
    p_fc.add_argument("--horizon", type=int, default=10, help="Forecast horizon (default: 10)")
    p_fc.add_argument(
        "--method",
        default="auto",
        choices=["auto", "arima", "sarima", "exp_smoothing"],
        help="Forecasting method (default: auto)",
    )
    p_fc.add_argument("--seasonal-period", type=int, default=None, help="Seasonal period (default: auto-detect)")
    p_fc.add_argument("--cv-splits", type=int, default=None, help="Walk-forward CV splits (optional)")
    p_fc.add_argument("--output", "-o", default=None, help="Output path (default: forecast.csv)")
    p_fc.add_argument("--plot", default=None, help="Save forecast plot as HTML")
    p_fc.add_argument("--silent", action="store_true", help="Suppress progress output")
    p_fc.set_defaults(func=cmd_forecast)

    # ── anomaly ──
    p_anom = subparsers.add_parser(
        "anomaly", help="Detect anomalies in tabular data", description="Multi-method consensus anomaly detection."
    )
    p_anom.add_argument("--data", required=True, help="Path to input data")
    p_anom.add_argument("--features", default=None, help="Comma-separated feature columns (default: all numeric)")
    p_anom.add_argument(
        "--methods",
        default="iforest,lof",
        help="Comma-separated methods: iforest,lof,tabnet,transformer (default: iforest,lof)",
    )
    p_anom.add_argument("--contamination", type=float, default=0.05, help="Expected anomaly fraction (default: 0.05)")
    p_anom.add_argument("--consensus", type=int, default=2, help="Min methods that must agree (default: 2)")
    p_anom.add_argument("--output", "-o", default=None, help="Output path (default: anomalies.csv)")
    p_anom.add_argument("--plot", default=None, help="Save anomaly plot as HTML")
    p_anom.add_argument("--silent", action="store_true", help="Suppress progress output")
    p_anom.set_defaults(func=cmd_anomaly)

    # ── fairness ──
    p_fair = subparsers.add_parser(
        "fairness", help="Check fairness and bias metrics", description="Compute fairness metrics on model predictions."
    )
    p_fair.add_argument("--data", required=True, help="Path to data with predictions")
    p_fair.add_argument("--target", required=True, help="True label column")
    p_fair.add_argument("--predicted", required=True, help="Predicted label column")
    p_fair.add_argument("--sensitive", required=True, help="Protected attribute column")
    p_fair.add_argument("--output", "-o", default=None, help="Output path for report JSON")
    p_fair.add_argument("--silent", action="store_true", help="Suppress progress output")
    p_fair.set_defaults(func=cmd_fairness)

    # ── compare ──
    p_cmp = subparsers.add_parser(
        "compare", help="Compare multiple model artifacts", description="Side-by-side comparison of .scomp artifacts."
    )
    p_cmp.add_argument("--artifacts", nargs="+", required=True, help="Paths to .scomp artifacts")
    p_cmp.add_argument("--output", "-o", default=None, help="Output path (default: stdout)")
    p_cmp.add_argument("--plot", default=None, help="Save comparison plot as HTML")
    p_cmp.set_defaults(func=cmd_compare)

    # ── init ──
    p_init = subparsers.add_parser(
        "init",
        help="Scaffold a new ML project",
        description="Generate a project directory with pipeline template, config, and folder structure.",
    )
    p_init.add_argument("name", help="Project name (creates a directory with this name)")
    p_init.add_argument("--force", action="store_true", help="Overwrite existing directory")
    p_init.set_defaults(func=cmd_init)

    # -- text --
    p_text = subparsers.add_parser(
        "text", help="Run text classification", description="Train a text classifier (TF-IDF or BERT contrastive)."
    )
    p_text.add_argument("--data", required=True, help="Path to input data")
    p_text.add_argument("--text-col", required=True, help="Column containing text data")
    p_text.add_argument("--target", required=True, help="Target label column")
    p_text.add_argument(
        "--method", choices=["tfidf", "contrastive"], default="tfidf", help="Classification method (default: tfidf)"
    )
    p_text.add_argument(
        "--model-name", default="bert-base-uncased", help="Transformer model name for contrastive method"
    )
    p_text.add_argument("--epochs", type=int, default=3, help="Training epochs (default: 3)")
    p_text.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    p_text.add_argument("--test-size", type=float, default=0.2, help="Test split (default: 0.2)")
    p_text.add_argument("--name", default=None, help="Pipeline name")
    p_text.add_argument("--output", "-o", default=None, help="Output path for results JSON")
    p_text.add_argument("--save-artifact", default=None, help="Save as .scomp artifact")
    p_text.add_argument("--silent", action="store_true", help="Suppress output")
    p_text.set_defaults(func=cmd_text)

    # -- cluster --
    p_clust = subparsers.add_parser(
        "cluster", help="Run clustering", description="Cluster data with KMeans or MeanShift."
    )
    p_clust.add_argument("--data", required=True, help="Path to input data")
    p_clust.add_argument("--features", default=None, help="Comma-separated feature columns (default: all numeric)")
    p_clust.add_argument("--n-clusters", type=int, default=5, help="Number of clusters (default: 5)")
    p_clust.add_argument(
        "--method", choices=["kmeans", "meanshift"], default="kmeans", help="Clustering method (default: kmeans)"
    )
    p_clust.add_argument("--name", default=None, help="Pipeline name")
    p_clust.add_argument("--output", "-o", default=None, help="Output CSV with cluster labels")
    p_clust.add_argument("--plot", default=None, help="Save cluster plot as HTML")
    p_clust.add_argument("--silent", action="store_true", help="Suppress output")
    p_clust.set_defaults(func=cmd_cluster)

    # -- tune --
    p_tune = subparsers.add_parser(
        "tune", help="Hyperparameter tuning", description="Optimize model hyperparameters with Optuna or HalvingSearch."
    )
    p_tune.add_argument("--data", required=True, help="Path to input data")
    p_tune.add_argument("--target", required=True, help="Target column")
    p_tune.add_argument("--task", required=True, choices=["regression", "classification"], help="ML task type")
    p_tune.add_argument(
        "--method", choices=["optuna", "halving"], default="optuna", help="Tuning method (default: optuna)"
    )
    p_tune.add_argument("--n-trials", type=int, default=50, help="Number of trials (default: 50)")
    p_tune.add_argument("--features", default=None, help="Comma-separated features (default: all)")
    p_tune.add_argument("--test-size", type=float, default=0.2, help="Test split (default: 0.2)")
    p_tune.add_argument("--output", "-o", default=None, help="Output path for results JSON")
    p_tune.add_argument("--save-artifact", default=None, help="Save best model as .scomp")
    p_tune.add_argument(
        "--format", choices=["json", "csv", "table"], default="json", help="Output format for results (default: json)"
    )
    p_tune.add_argument("--silent", action="store_true", help="Suppress output")
    p_tune.set_defaults(func=cmd_tune)

    # -- validate --
    p_val = subparsers.add_parser(
        "validate",
        help="Validate a model on test data",
        description="Evaluate a saved artifact on new data with metrics and report.",
    )
    p_val.add_argument("--artifact", required=True, help="Path to .scomp artifact")
    p_val.add_argument("--data", required=True, help="Path to test data")
    p_val.add_argument("--target", required=True, help="Target column in test data")
    p_val.add_argument("--output", "-o", default=None, help="Output path for metrics JSON")
    p_val.add_argument("--report", default=None, help="Generate HTML validation report")
    p_val.add_argument(
        "--format", choices=["json", "csv", "table"], default="json", help="Output format for metrics (default: json)"
    )
    p_val.add_argument("--silent", action="store_true", help="Suppress output")
    p_val.set_defaults(func=cmd_validate)

    # -- monitor --
    p_mon = subparsers.add_parser(
        "monitor", help="Production monitoring report", description="Combined drift + quality + performance check."
    )
    p_mon.add_argument("--reference", required=True, help="Path to reference (training) data")
    p_mon.add_argument("--current", required=True, help="Path to current (production) data")
    p_mon.add_argument("--artifact", default=None, help="Path to .scomp artifact (for performance eval)")
    p_mon.add_argument("--target", default=None, help="Target column (needed with --artifact)")
    p_mon.add_argument("--threshold", type=float, default=0.2, help="PSI drift threshold (default: 0.2)")
    p_mon.add_argument("--output", "-o", default=None, help="Output HTML report path")
    p_mon.add_argument("--silent", action="store_true", help="Suppress output")
    p_mon.set_defaults(func=cmd_monitor)

    # -- list-models --
    p_lm = subparsers.add_parser("list-models", help="List available model types")
    p_lm.set_defaults(func=cmd_list_models)

    # -- check-deps --
    p_cd = subparsers.add_parser("check-deps", help="Check installed optional dependencies")
    p_cd.set_defaults(func=cmd_check_deps)

    # -- mcp --
    p_mcp = subparsers.add_parser(
        "mcp",
        help="Start MCP server for AI agent integration",
        description="Start the Model Context Protocol server (stdio transport). "
        "Compatible with Claude Desktop, Kiro, Cursor, VS Code Copilot, "
        "and any MCP-compatible AI agent. Exposes 15 ML tools, "
        "3 data resources, and 4 workflow prompts.",
    )
    p_mcp.set_defaults(func=cmd_mcp)

    # -- serve --
    p_serve = subparsers.add_parser(
        "serve",
        help="Serve a model as REST API",
        description="Start a Flask server exposing a .scomp artifact for predictions via HTTP. "
        "Endpoints: GET /health, GET /info, GET /schema, POST /predict.",
    )
    p_serve.add_argument("--artifact", required=True, help="Path to .scomp artifact to serve")
    p_serve.add_argument("--host", default="0.0.0.0", help="Bind address (default: 0.0.0.0)")
    p_serve.add_argument("--port", type=int, default=8080, help="Port number (default: 8080)")
    p_serve.add_argument("--debug", action="store_true", help="Enable Flask debug mode")
    p_serve.set_defaults(func=cmd_serve)

    # -- pipeline --
    p_pipe = subparsers.add_parser(
        "pipeline",
        help="Run pipeline from YAML config",
        description="Execute a multi-step ML pipeline defined in a YAML file. "
        "Steps: load data -> engineer features -> tune/train -> validate -> save artifact. "
        "All steps except data/target/task are optional.",
    )
    p_pipe.add_argument("--config", required=True, help="Path to YAML pipeline config file")
    p_pipe.add_argument("--silent", action="store_true", help="Suppress progress output")
    p_pipe.set_defaults(func=cmd_pipeline)

    # -- describe --
    p_desc = subparsers.add_parser(
        "describe",
        help="Quick dataset profiling",
        description="Print a summary table with one row per column: dtype, missing%%, "
        "unique count, min, max, mean, std. Faster and lighter than 'quality'.",
    )
    p_desc.add_argument("--data", required=True, help="Path to dataset (CSV, TSV, Parquet)")
    p_desc.add_argument("--output", "-o", default=None, help="Save output to file")
    p_desc.add_argument(
        "--format", choices=["table", "csv", "json"], default="table", help="Output format (default: table)"
    )
    p_desc.set_defaults(func=cmd_describe)

    # -- export --
    p_exp2 = subparsers.add_parser(
        "export",
        help="Export model to standard format",
        description="Convert a .scomp artifact to pickle, ONNX, PMML, or joblib format "
        "for deployment in external systems.",
    )
    p_exp2.add_argument("--artifact", required=True, help="Path to .scomp artifact")
    p_exp2.add_argument(
        "--format",
        choices=["pickle", "onnx", "pmml", "joblib"],
        default="pickle",
        help="Export format (default: pickle)",
    )
    p_exp2.add_argument("--output", "-o", default=None, help="Output file path (default: auto-named)")
    p_exp2.set_defaults(func=cmd_export)

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
