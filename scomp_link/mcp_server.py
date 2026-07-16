# -*- coding: utf-8 -*-
"""
███╗   ███╗ █████╗ ██████╗     ██████╗ ███████╗██████╗ ██╗   ██╗███████╗██████╗
████╗ ████║██╔══██╗██╔══██╗    ██╔════╝██╔════╝██╔══██╗██║   ██║██╔════╝██╔══██╗
██╔████╔██║██║  ╚═╝██████╔╝    ╚█████╗ █████╗  ██████╔╝╚██╗ ██╔╝█████╗  ██████╔╝
██║╚██╔╝██║██║  ██╗██╔═══╝      ╚═══██╗██╔══╝  ██╔══██╗ ╚████╔╝ ██╔══╝  ██╔══██╗
██║ ╚═╝ ██║╚█████╔╝██║         ██████╔╝███████╗██║  ██║  ╚██╔╝  ███████╗██║  ██║
╚═╝     ╚═╝ ╚════╝ ╚═╝         ╚═════╝ ╚══════╝╚═╝  ╚═╝   ╚═╝   ╚══════╝╚═╝  ╚═╝

MCP (Model Context Protocol) server for scomp-link.
Exposes ML toolkit functionality as tools, resources, and prompts
for AI agents (Claude, Kiro, Cursor, VS Code Copilot, etc.)

Usage:
    scomp-link mcp          # Start via CLI
    python -m scomp_link.mcp_server   # Start directly
"""

import json
import os
from typing import Optional

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "scomp-link",
    instructions="End-to-end ML toolkit: train models, tune hyperparameters, detect drift, "
    "generate HTML reports with 39 chart types, detect anomalies, forecast time series, "
    "check fairness, and serve models as REST APIs.",
)


# ═══════════════════════════════════════════════════════════════════
# TOOLS
# ═══════════════════════════════════════════════════════════════════


@mcp.tool()
def describe_data(path: str) -> str:
    """Profile a dataset. Returns column-level stats: dtype, missing%, unique count, min, max, mean, std.
    Use this first to understand any new dataset before training or visualization."""
    import pandas as pd

    df = _load_df(path)
    rows = []
    for col in df.columns:
        row = {
            "column": col,
            "dtype": str(df[col].dtype),
            "missing_pct": round(df[col].isnull().mean() * 100, 1),
            "unique": int(df[col].nunique()),
        }
        if pd.api.types.is_numeric_dtype(df[col]):
            row.update(
                {
                    "min": round(float(df[col].min()), 4),
                    "max": round(float(df[col].max()), 4),
                    "mean": round(float(df[col].mean()), 4),
                    "std": round(float(df[col].std()), 4),
                }
            )
        rows.append(row)
    return json.dumps({"shape": list(df.shape), "columns": rows}, indent=2)


@mcp.tool()
def train_model(
    data: str,
    target: str,
    task: str = "regression",
    engineer: bool = False,
    tune: bool = False,
    n_trials: int = 50,
    save_artifact: Optional[str] = None,
) -> str:
    """Train an ML model. Supports regression, classification, text, clustering.
    Set engineer=true for automatic feature engineering. Set tune=true for Optuna hyperparameter optimization.
    Returns metrics and optional artifact path."""
    import scomp_link

    scomp_link.set_verbosity("silent")
    import pandas as pd

    df = _load_df(data)
    if engineer:
        fe = scomp_link.FeatureEngineer(interactions=True, log_transform=True)
        y = df[target]
        X = fe.fit_transform(df.drop(columns=[target]), y)
        df = X.copy()
        df[target] = y.values

    if tune:
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
        from sklearn.model_selection import train_test_split

        from scomp_link.models.advanced_tuning import OptunaOptimizer

        X = df.drop(columns=[target])
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        base = GradientBoostingRegressor if task == "regression" else GradientBoostingClassifier
        scoring = "r2" if task == "regression" else "accuracy"

        def param_space(trial):
            return {
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "max_depth": trial.suggest_int("max_depth", 3, 12),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            }

        opt = OptunaOptimizer(base, param_space, scoring=scoring, n_trials=n_trials)
        best_model = opt.optimize(X_train, y_train)

        y_pred = best_model.predict(X_test)
        if task == "regression":
            from sklearn.metrics import mean_squared_error, r2_score

            metrics = {
                "r2": round(r2_score(y_test, y_pred), 4),
                "rmse": round(mean_squared_error(y_test, y_pred) ** 0.5, 4),
            }
        else:
            from sklearn.metrics import accuracy_score, f1_score

            metrics = {
                "accuracy": round(accuracy_score(y_test, y_pred), 4),
                "f1": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4),
            }

        result = {"status": "success", "model_type": type(best_model).__name__, "metrics": metrics}
        if save_artifact:
            artifact = scomp_link.ScompArtifact()
            artifact.set_model(best_model)
            artifact.set_config(task_type=task, target_col=target)
            artifact.set_metrics(metrics)
            artifact.set_feature_schema(X_train)
            artifact.save(save_artifact)
            result["artifact_path"] = save_artifact
    else:
        pipe = scomp_link.ScompLinkPipeline("mcp_train")
        pipe.import_and_clean_data(df)
        pipe.select_variables(target_col=target)
        pipe.choose_model("numerical_prediction" if task == "regression" else "categorical_known")
        results = pipe.run_pipeline(task_type=task)
        result = {
            "status": results.get("status", "success"),
            "model_type": results.get("model_type"),
            "metrics": results.get("metrics"),
        }
        if save_artifact:
            artifact = scomp_link.ScompArtifact()
            artifact.set_model(pipe.model)
            artifact.set_config(task_type=task, target_col=target)
            artifact.set_metrics(results.get("metrics", {}))
            artifact.save(save_artifact)
            result["artifact_path"] = save_artifact

    return json.dumps(result, indent=2, default=str)


@mcp.tool()
def predict(artifact: str, data: str, output: Optional[str] = None) -> str:
    """Generate predictions from a .scomp artifact on new data.
    Returns predictions array. Optionally saves to CSV."""
    import pandas as pd

    import scomp_link

    scomp_link.set_verbosity("silent")

    loaded = scomp_link.ScompArtifact.load(artifact)
    df = _load_df(data)
    target_col = loaded.config.get("target_col")
    feature_cols = [c for c in df.columns if c != target_col]
    predictions = loaded.predict(df[feature_cols])

    if output:
        out_df = df.copy()
        out_df["prediction"] = predictions
        out_df.to_csv(output, index=False)

    preds_list = predictions.tolist() if hasattr(predictions, "tolist") else list(predictions)
    return json.dumps(
        {"n_predictions": len(preds_list), "predictions": preds_list[:20], "output_path": output}, default=str
    )


@mcp.tool()
def validate_model(artifact: str, data: str, target: str, report: Optional[str] = None) -> str:
    """Evaluate a saved model on test data. Returns metrics (regression: r2, rmse, mae; classification: accuracy, f1, precision, recall).
    Optionally generates an HTML validation report."""
    import scomp_link

    scomp_link.set_verbosity("silent")

    loaded = scomp_link.ScompArtifact.load(artifact)
    df = _load_df(data)
    feature_cols = [c for c in df.columns if c != target]
    X, y = df[feature_cols], df[target]
    predictions = loaded.predict(X)

    task_type = loaded.config.get("task_type", "regression")
    validator = scomp_link.Validator(loaded.model)
    metrics = validator.evaluate(y, predictions, task_type=task_type)

    if report:
        validator.generate_validation_report(y, predictions, task_type=task_type, report_name=report)

    return json.dumps(
        {"task_type": task_type, "metrics": metrics, "n_samples": len(y), "report_path": report}, indent=2, default=str
    )


@mcp.tool()
def detect_drift(reference: str, current: str, threshold: float = 0.2, plot: Optional[str] = None) -> str:
    """Detect distribution drift between reference (training) and current (production) data.
    Returns PSI per feature and which features drifted."""
    import scomp_link

    scomp_link.set_verbosity("silent")

    df_ref = _load_df(reference)
    df_cur = _load_df(current)
    numeric_cols = df_ref.select_dtypes(include=["number"]).columns.tolist()
    common = [c for c in numeric_cols if c in df_cur.columns]

    detector = scomp_link.DriftDetector(df_ref[common], psi_threshold=threshold)
    report = detector.detect(df_cur[common])
    summary = detector.summary(report)

    if plot:
        fig = detector.plot_drift_report(report)
        fig.write_html(plot)

    return json.dumps(
        {
            "drifted_features": summary["drifted_features"],
            "total_features": summary["total_features"],
            "worst_feature": summary.get("worst_feature"),
            "max_psi": round(summary.get("max_psi", 0), 4),
            "plot_path": plot,
        },
        indent=2,
        default=str,
    )


@mcp.tool()
def detect_anomalies(data: str, methods: str = "iforest,lof", contamination: float = 0.05, consensus: int = 2) -> str:
    """Detect anomalies using multi-method consensus (Isolation Forest, LOF, TabNet, Transformer).
    Returns number of anomalies and per-method comparison."""
    import scomp_link

    scomp_link.set_verbosity("silent")

    df = _load_df(data)
    features = df.select_dtypes(include=["number"]).columns.tolist()
    method_list = [m.strip() for m in methods.split(",")]

    detector = scomp_link.AnomalyDetector(
        contamination=contamination, methods=method_list, consensus_threshold=consensus, verbose=False
    )
    results = detector.fit_predict(df, features=features)

    comparison = results["comparison"].to_dict("records")
    n_anomalies = int(results["data"]["is_anomaly"].sum())

    return json.dumps({"n_anomalies": n_anomalies, "total_rows": len(df), "methods": comparison}, indent=2, default=str)


@mcp.tool()
def check_fairness(data: str, target: str, predicted: str, sensitive: str) -> str:
    """Compute fairness metrics: demographic parity, disparate impact (4/5 rule), equalized odds.
    Returns whether the model is fair and detailed group-level metrics."""
    import scomp_link

    scomp_link.set_verbosity("silent")

    df = _load_df(data)
    fm = scomp_link.FairnessMetrics(df[target].values, df[predicted].values, sensitive_feature=df[sensitive].values)
    report = fm.compute_all()

    return json.dumps(
        {
            "demographic_parity": report["demographic_parity"],
            "disparate_impact": report["disparate_impact"],
            "equalized_odds": {
                "tpr_diff": report["equalized_odds"]["tpr_diff"],
                "fpr_diff": report["equalized_odds"]["fpr_diff"],
                "fair": report["equalized_odds"]["fair"],
            },
        },
        indent=2,
        default=str,
    )


@mcp.tool()
def forecast_series(data: str, column: str, horizon: int = 10, method: str = "auto", plot: Optional[str] = None) -> str:
    """Forecast future values of a time series column. Methods: auto, arima, exp_smoothing.
    Returns predicted values and optional plot."""
    import scomp_link

    scomp_link.set_verbosity("silent")

    df = _load_df(data)
    series = df[column].dropna()

    fc = scomp_link.TimeSeriesForecaster(method=method, horizon=horizon)
    fc.fit(series)
    ci = fc.predict_with_ci(steps=horizon)

    if plot:
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=series.values, name="Historical"))
        fig.add_trace(
            go.Scatter(
                x=list(range(len(series), len(series) + horizon)),
                y=ci["forecast"].values,
                name="Forecast",
                line=dict(dash="dash"),
            )
        )
        fig.update_layout(title=f"Forecast: {column}")
        fig.write_html(plot)

    return json.dumps(
        {"horizon": horizon, "method": method, "forecast": ci["forecast"].round(4).tolist(), "plot_path": plot},
        indent=2,
        default=str,
    )


@mcp.tool()
def engineer_features(
    data: str, target: str, interactions: bool = True, log_transform: bool = True, output: Optional[str] = None
) -> str:
    """Apply automated feature engineering: polynomial interactions, log transforms for skewed features,
    date extraction, target encoding. Returns output path and new shape."""
    import pandas as pd

    import scomp_link

    scomp_link.set_verbosity("silent")

    df = _load_df(data)
    y = df[target]
    X = df.drop(columns=[target])

    fe = scomp_link.FeatureEngineer(interactions=interactions, log_transform=log_transform)
    X_eng = fe.fit_transform(X, y)
    X_eng[target] = y.values

    out_path = output or data.replace(".csv", "_engineered.csv")
    X_eng.to_csv(out_path, index=False)

    return json.dumps(
        {
            "output_path": out_path,
            "original_shape": list(df.shape),
            "engineered_shape": list(X_eng.shape),
            "new_columns": [c for c in X_eng.columns if c not in df.columns],
        },
        indent=2,
    )


@mcp.tool()
def cluster_data(
    data: str, n_clusters: int = 5, method: str = "kmeans", features: Optional[str] = None, output: Optional[str] = None
) -> str:
    """Cluster data using KMeans or MeanShift. Returns cluster labels and silhouette score.
    Optionally saves result with cluster column."""
    import numpy as np
    import pandas as pd
    from sklearn.metrics import silhouette_score

    import scomp_link

    scomp_link.set_verbosity("silent")

    df = _load_df(data)
    feat_cols = features.split(",") if features else df.select_dtypes(include=["number"]).columns.tolist()
    X = df[feat_cols].values

    if method == "kmeans":
        from sklearn.cluster import KMeans

        model = KMeans(n_clusters=n_clusters, random_state=42)
    else:
        from sklearn.cluster import MeanShift

        model = MeanShift()

    labels = model.fit_predict(X)
    sil = float(silhouette_score(X, labels))

    if output:
        out_df = df.copy()
        out_df["cluster"] = labels
        out_df.to_csv(output, index=False)

    return json.dumps(
        {
            "n_clusters": int(len(np.unique(labels))),
            "silhouette_score": round(sil, 4),
            "cluster_sizes": {str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
            "output_path": output,
        },
        indent=2,
    )


@mcp.tool()
def generate_report(
    data: str,
    output: str = "report.html",
    title: str = "Analysis Report",
    artifact: Optional[str] = None,
    footer_html: Optional[str] = None,
) -> str:
    """Generate an interactive HTML report. If only data is provided, creates an EDA report.
    If artifact is also provided, creates a model evaluation report with predictions and SHAP."""
    import pandas as pd

    import scomp_link

    scomp_link.set_verbosity("silent")

    from scomp_link.config import get_report_defaults
    from scomp_link.utils.plotly_utils import barchart, histogram
    from scomp_link.utils.report_html import ScompLinkHTMLReport

    defaults = get_report_defaults()
    effective_footer = footer_html if footer_html is not None else defaults.get("footer_html")
    df = _load_df(data)
    report = ScompLinkHTMLReport(title=title, footer_html=effective_footer)

    # Overview section
    report.open_section("Dataset Overview")
    overview = pd.DataFrame(
        [
            {
                "rows": len(df),
                "columns": len(df.columns),
                "numeric": len(df.select_dtypes(include=["number"]).columns),
                "categorical": len(df.select_dtypes(include=["object"]).columns),
                "missing_total": int(df.isnull().sum().sum()),
            }
        ]
    )
    report.add_dataframe(overview, "Overview")
    report.close_section()

    # Distributions
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        report.open_section("Feature Distributions")
        for col in numeric_cols[:8]:
            fig = histogram(df[col].dropna().values, f"{col}")
            report.add_graph_to_report(fig, col)
        report.close_section()

    # Model evaluation if artifact provided
    if artifact and os.path.exists(artifact):
        loaded = scomp_link.ScompArtifact.load(artifact)
        target_col = loaded.config.get("target_col")
        if target_col and target_col in df.columns:
            report.open_section("Model Performance")
            feature_cols = [c for c in df.columns if c != target_col]
            preds = loaded.predict(df[feature_cols])
            metrics = scomp_link.Validator(loaded.model).evaluate(
                df[target_col], preds, task_type=loaded.config.get("task_type", "regression")
            )
            report.add_dataframe(pd.DataFrame([metrics]), "Metrics")
            report.close_section()

    report.save_html(output)
    size_kb = os.path.getsize(output) / 1024
    return json.dumps({"report_path": output, "size_kb": round(size_kb, 1)}, indent=2)


@mcp.tool()
def create_visualization(
    chart_type: str, data: str, title: str = "Chart", output: Optional[str] = None, **kwargs
) -> str:
    """Create a single visualization. chart_type can be any of the 39 available charts.
    Categories: plotly (histogram, barchart, linechart, area_chart), rawgraphs (treemap, sankey, sunburst, etc.),
    highcharts (streamgraphs, calendar_heatmap, calendar_gantt).
    data should be a JSON string with the chart-specific data format."""
    import json as json_mod

    chart_data = json_mod.loads(data) if isinstance(data, str) else data

    if chart_type in ("histogram", "barchart", "linechart", "area_chart"):
        from scomp_link.utils import plotly_utils

        func = getattr(plotly_utils, chart_type)
        if chart_type == "histogram":
            fig = func(chart_data["values"], title)
        elif chart_type == "barchart":
            fig = func(chart_data["categories"], chart_data["values"], title)
        else:
            fig = func(chart_data["dates"], chart_data["lines"], title)
        if output:
            fig.write_html(output)
        return json.dumps({"type": chart_type, "engine": "plotly", "output": output})

    elif chart_type in ("streamgraphs", "calendar_heatmap", "calendar_gantt"):
        from scomp_link.utils import highcharts

        func = getattr(highcharts, chart_type)
        html = func(title, **chart_data)
        if output:
            with open(output, "w") as f:
                f.write(
                    f"<html><head><script src='https://code.highcharts.com/highcharts.js'></script></head><body>{html}</body></html>"
                )
        return json.dumps({"type": chart_type, "engine": "highcharts", "output": output})

    else:
        from scomp_link.utils import rawgraphs

        func = getattr(rawgraphs, chart_type, None)
        if func is None:
            return json.dumps({"error": f"Unknown chart type: {chart_type}"})
        svg = func(**chart_data, title=title)
        if output:
            with open(output, "w") as f:
                f.write(svg)
        return json.dumps({"type": chart_type, "engine": "rawgraphs", "output": output})


@mcp.tool()
def compare_models(artifacts: str, plot: Optional[str] = None) -> str:
    """Compare multiple .scomp artifacts side by side. artifacts is a comma-separated list of paths.
    Returns a comparison table of metrics."""
    import pandas as pd

    import scomp_link

    scomp_link.set_verbosity("silent")

    paths = [p.strip() for p in artifacts.split(",")]
    rows = []
    for path in paths:
        a = scomp_link.ScompArtifact.load(path)
        row = {"artifact": os.path.basename(path), "model_type": type(a.model).__name__}
        row.update(a.metrics)
        rows.append(row)

    comparison = pd.DataFrame(rows)

    if plot:
        import plotly.express as px

        numeric_cols = comparison.select_dtypes(include=["number"]).columns.tolist()
        if numeric_cols:
            fig = px.bar(comparison, x="artifact", y=numeric_cols, barmode="group", title="Model Comparison")
            fig.write_html(plot)

    return json.dumps({"comparison": rows, "plot_path": plot}, indent=2, default=str)


@mcp.tool()
def export_model(artifact: str, format: str = "pickle", output: Optional[str] = None) -> str:
    """Export a .scomp model to standard format: pickle, joblib, onnx.
    Returns the output file path and size."""
    from pathlib import Path

    import scomp_link

    scomp_link.set_verbosity("silent")

    loaded = scomp_link.ScompArtifact.load(artifact)
    out_path = output or artifact.replace(".scomp", f".{format}")

    if format == "pickle":
        import pickle

        with open(out_path, "wb") as f:
            pickle.dump(loaded.model, f)
    elif format == "joblib":
        import joblib

        joblib.dump(loaded.model, out_path)
    else:
        return json.dumps({"error": f"Format {format} requires additional deps (skl2onnx/sklearn2pmml)"})

    size_kb = Path(out_path).stat().st_size / 1024
    return json.dumps({"output_path": out_path, "format": format, "size_kb": round(size_kb, 1)})


@mcp.tool()
def embed_text(artifact: str, data: str, text_col: str = "text", output: Optional[str] = None) -> str:
    """Generate embeddings from a trained contrastive text model (.scomp artifact).
    Returns embedding shape and optionally saves to .npy file.
    Use this to extract text representations for downstream analysis."""
    import numpy as np

    import scomp_link

    scomp_link.set_verbosity("silent")

    loaded = scomp_link.ScompArtifact.load(artifact)
    if not hasattr(loaded.model, "embed"):
        return json.dumps({"error": "Artifact is not a contrastive text model"})

    df = _load_df(data)
    if text_col not in df.columns:
        return json.dumps({"error": f"Column '{text_col}' not found"})

    texts = df[text_col].tolist()
    embeddings = loaded.model.embed(texts)

    result = {"shape": list(embeddings.shape), "dtype": str(embeddings.dtype)}
    if output:
        if output.endswith(".npy"):
            np.save(output, embeddings)
        else:
            import pandas as pd

            pd.DataFrame(embeddings).to_csv(output, index=False)
        result["output_path"] = output

    return json.dumps(result, indent=2)


@mcp.tool()
def select_backbone(
    data: str,
    text_col: str = "text",
    label_col: str = "label",
    candidates: Optional[str] = None,
    sample_size: int = 500,
) -> str:
    """Find the best pretrained embedding backbone for a dataset.
    Evaluates multiple sentence-transformer models by computing contrastive loss
    on a sample. Returns ranked list of models with loss scores.
    Use this before training to choose the optimal starting point."""
    from scomp_link.models.contrastive_text import EmbeddingSelector

    df = _load_df(data)
    if text_col not in df.columns or label_col not in df.columns:
        return json.dumps({"error": f"Columns not found. Available: {list(df.columns)}"})

    candidate_list = candidates.split(",") if candidates else None
    selector = EmbeddingSelector(candidates=candidate_list)
    results = selector.find_best_backbone(df, text_col=text_col, label_col=label_col, sample_size=sample_size)

    return json.dumps(
        {
            "best_model": results.iloc[0]["model"],
            "best_loss": float(results.iloc[0]["loss"]),
            "ranking": results.to_dict("records"),
        },
        indent=2,
        default=str,
    )


# ═══════════════════════════════════════════════════════════════════
# REPORT BUILDER TOOLS (stateful)
# ═══════════════════════════════════════════════════════════════════

_reports: dict = {}


def _get_report(report_id: str):
    """Get a report from the store or return None."""
    return _reports.get(report_id)


@mcp.tool()
def report_create(
    title: str,
    font_family: Optional[str] = None,
    url_img_logo: Optional[str] = None,
    url_background_header: Optional[str] = None,
    description: Optional[str] = None,
    author: Optional[str] = None,
    language: Optional[str] = None,
    main_color: Optional[str] = None,
    light_color: Optional[str] = None,
    dark_color: Optional[str] = None,
    footer_html: Optional[str] = None,
) -> str:
    """Create a new report session. Returns a report_id to use with other report_* tools.

    All parameters (except title) default to values from config files:
    - ~/.scomp-link/config.yaml (global defaults, e.g. corporate branding)
    - .scomp-link.yaml (project-local overrides)
    If no config file exists, scomp-link built-in defaults are used.

    Parameters:
        title: Report title displayed in the header.
        font_family: CSS font family (e.g. "Arial", "Baloo 2", "Roboto").
        url_img_logo: Public URL to a logo image (used as favicon). Empty string = no logo.
        url_background_header: URL to header background image. Ideal size: 1920x600px.
        description: HTML meta description for the report.
        author: HTML meta author.
        language: HTML lang attribute (e.g. "en", "it", "de").
        main_color: Primary accent color as hex (e.g. "#CC0000"). Used for headings, links, buttons.
        light_color: Light variant of accent (e.g. "#FF3333"). Used for hover states.
        dark_color: Dark variant of accent (e.g. "#990000"). Used for borders, button outlines.
        footer_html: Custom HTML wrapped in <footer>...</footer>. If null, shows scomp-link default.

    Example workflow:
        1. report_create("Q4 Sales Analysis") → returns report_id
        2. report_add_section(report_id, "Overview")
        3. report_add_chart(report_id, "plotly", "barchart", data, "Revenue by Region")
        4. report_add_table(report_id, data, "Top 10 Products")
        5. report_save(report_id, "q4_report.html")
    """
    import uuid

    from scomp_link.config import get_report_defaults
    from scomp_link.utils.report_html import ScompLinkHTMLReport

    defaults = get_report_defaults()
    params = {
        "title": title,
        "font_family": font_family or defaults.get("font_family", "Baloo 2"),
        "url_img_logo": url_img_logo if url_img_logo is not None else defaults.get("url_img_logo", ""),
        "url_background_header": url_background_header or defaults.get("url_background_header", ""),
        "description": description or defaults.get("description", "Automatic Report"),
        "author": author or defaults.get("author", "scomp-link toolkit"),
        "language": language or defaults.get("language", "en"),
        "main_color": main_color or defaults.get("main_color", "#6E37FA"),
        "light_color": light_color or defaults.get("light_color", "#9682FF"),
        "dark_color": dark_color or defaults.get("dark_color", "#4614B4"),
        "footer_html": footer_html if footer_html is not None else defaults.get("footer_html"),
    }

    report = ScompLinkHTMLReport(**params)
    report_id = uuid.uuid4().hex[:8]
    _reports[report_id] = report

    return json.dumps(
        {"report_id": report_id, "title": title, "params": {k: v for k, v in params.items() if k != "title"}}, indent=2
    )


@mcp.tool()
def report_add_section(report_id: str, title: str) -> str:
    """Open a new collapsible section in the report. Automatically closes the previous section if one is open.

    Parameters:
        report_id: The report ID returned by report_create.
        title: Section heading text.
    """
    report = _get_report(report_id)
    if not report:
        return json.dumps({"error": f"Report '{report_id}' not found. Use report_create first."})

    if report.section_just_open:
        report.close_section()
    report.open_section(title)
    return json.dumps({"status": "section_opened", "title": title})


@mcp.tool()
def report_add_text(report_id: str, content: str, style: str = "paragraph") -> str:
    """Add text content to the current section of a report.

    Parameters:
        report_id: The report ID returned by report_create.
        content: The text or HTML content to add.
        style: How to render the content:
            - "paragraph": Plain text paragraph (default)
            - "title": Large heading (h2 style)
            - "subtitle": Medium heading (h3 style)
            - "html": Raw HTML inserted as-is (for custom formatting)
    """
    report = _get_report(report_id)
    if not report:
        return json.dumps({"error": f"Report '{report_id}' not found. Use report_create first."})

    if style == "title":
        report.add_title(content)
    elif style == "subtitle":
        report.html_report += f"<h3>{content}</h3>"
    elif style == "html":
        report.html_report += content
    else:
        report.add_text(content)

    return json.dumps({"status": "text_added", "style": style, "length": len(content)})


@mcp.tool()
def report_add_table(report_id: str, data: str, title: str = "Table") -> str:
    """Add a data table to the report. The table is interactive (sortable, CSV-exportable).

    Parameters:
        report_id: The report ID returned by report_create.
        data: JSON string — a list of dictionaries (one per row).
              Example: '[{"name": "Alice", "score": 95}, {"name": "Bob", "score": 87}]'
        title: Table caption/title.
    """
    import pandas as pd

    report = _get_report(report_id)
    if not report:
        return json.dumps({"error": f"Report '{report_id}' not found. Use report_create first."})

    rows = json.loads(data) if isinstance(data, str) else data
    df = pd.DataFrame(rows)
    report.add_dataframe(df, title)
    return json.dumps({"status": "table_added", "title": title, "rows": len(df), "columns": len(df.columns)})


@mcp.tool()
def report_add_chart(report_id: str, engine: str, chart_type: str, data: str, title: str = "Chart") -> str:
    """Add a chart to an open report. Supports 3 engines and 39 chart types.

    Parameters:
        report_id: The report ID returned by report_create.
        engine: Chart rendering engine — "plotly", "rawgraphs", or "highcharts".
        chart_type: The specific chart type (see tables below).
        data: JSON string with chart-specific data format (see examples below).
        title: Chart title displayed above the visualization.

    ═══════════════════════════════════════════════════════════════════════════
    ENGINE SELECTION GUIDE:
    ═══════════════════════════════════════════════════════════════════════════
    • "plotly"     → Interactive (hover, zoom, pan). Best for dashboards & exploration.
    • "rawgraphs"  → Static SVG, publication-quality. Best for print, PDF, presentations.
    • "highcharts" → Interactive time series & calendars. Best for monitoring dashboards.

    ═══════════════════════════════════════════════════════════════════════════
    PLOTLY CHARTS (4 types — interactive with hover/zoom):
    ═══════════════════════════════════════════════════════════════════════════
    histogram   │ {"values": [1.2, 3.4, 5.6, ...], "name": "column_name"}
    barchart    │ {"categories": ["A","B","C"], "values": [[10, 20, 30]],
                │  "y_axis_titles": ["Revenue"]}
                │  For multiple series: "values": [[10,20,30], [5,15,25]],
                │                       "y_axis_titles": ["Revenue", "Cost"]
    linechart   │ {"dates": ["2024-01-01","2024-02-01",...],
                │  "lines": [[100, 120, 140,...]], "y_labels": ["Sales"]}
                │  Multiple lines: "lines": [[...], [...]], "y_labels": ["A","B"]
    area_chart  │ Same format as linechart (stacked area)

    ═══════════════════════════════════════════════════════════════════════════
    RAWGRAPHS CHARTS (31 types — static SVG, publication-quality):
    ═══════════════════════════════════════════════════════════════════════════
    COMPARISONS:
    barchart         │ {"categories": ["Q1","Q2"], "values": [120, 180]}
    barchartmultiset │ {"categories": ["Q1","Q2"], "groups": {"Product A": [120,150], "Product B": [80,90]}}
    barchartstacked  │ {"categories": ["Q1","Q2"], "groups": {"Product A": [120,150], "Product B": [80,90]}}
    piechart         │ {"labels": ["Segment A","Segment B"], "values": [60, 40]}
    radarchart       │ {"categories": ["Speed","Power","Range"], "series": {"Model X": [8,6,9], "Model Y": [7,8,6]}}
    voronoidiagram   │ {"points": [[1.0, 2.0], [3.0, 4.0], [5.0, 1.0]]}

    DISTRIBUTIONS:
    beeswarm         │ {"data": [1.2, 3.4, 2.1, ...], "groups": ["A","A","B",...]}
    boxplot          │ {"data": [[1,2,3,4,5], [2,3,4,5,6]], "labels": ["Group A","Group B"]}
    violinplot       │ {"data": [[1,2,3,4,5], [2,3,4,5,6]], "labels": ["Group A","Group B"]}

    TIME SERIES:
    bumpchart        │ {"ranks": {"Team A": [1,2,1,3], "Team B": [2,1,3,1]}, "periods": ["Q1","Q2","Q3","Q4"]}
    gantt_chart      │ {"tasks": [{"name": "Design", "start": "2024-01-01", "end": "2024-02-01", "group": "Phase 1"}]}
    horizongraph     │ {"series": {"Temp": [20,22,19,25,23]}, "x_values": ["Mon","Tue","Wed","Thu","Fri"]}
    linechart        │ {"series": {"Revenue": [100,120,140]}, "x_values": ["Jan","Feb","Mar"]}
    slopechart       │ {"data": {"Product A": [100, 150], "Product B": [120, 90]}}
    streamgraph      │ {"series": {"Cat A": [10,20,15], "Cat B": [5,15,20]}, "x_values": ["Jan","Feb","Mar"]}

    CORRELATIONS:
    bubblechart      │ {"x": [1,2,3], "y": [4,5,6], "size": [10,20,30]}
                     │  Optional: "labels": ["A","B","C"], "groups": ["G1","G1","G2"]
    contour_plot     │ {"x": [1,2,3,...], "y": [4,5,6,...]}
    convex_hull      │ {"x": [1,2,3,...], "y": [4,5,6,...], "groups": ["A","A","B",...]}
    hexagonal_binning│ {"x": [1,2,3,...], "y": [4,5,6,...]}
    matrixplot       │ {"matrix": [[1,0.5],[0.5,1]], "row_labels": ["A","B"], "col_labels": ["A","B"]}
    parallelcoordinates │ {"data": {"speed": [1,2,3], "power": [4,5,6]}, "class_column": "speed"}

    HIERARCHIES (nested data format):
    circlepacking       │ {"data": {"name": "root", "children": [{"name": "A", "value": 30}, {"name": "B", "value": 20}]}}
    circular_dendrogram │ {"linkage_matrix": [[0,1,0.5,2],[2,3,1.0,3]], "labels": ["a","b","c","d"]}
    dendrogram          │ {"linkage_matrix": [[0,1,0.5,2],[2,3,1.0,3]], "labels": ["a","b","c","d"]}
    sunburst            │ {"data": {"name": "root", "children": [{"name": "A", "children": [{"name": "A1", "value": 10}]}]}}
    treemap             │ {"data": {"name": "root", "children": [{"name": "Division A", "value": 100}, {"name": "Division B", "value": 60}]}}
    voronoi_treemap     │ {"data": {"name": "root", "children": [{"name": "A", "value": 30}]}}

    NETWORKS:
    alluvial_diagram │ {"flows": [{"source": "Homepage", "target": "Products", "value": 40}]}
    arc_diagram      │ {"nodes": ["A","B","C"], "links": [{"source": 0, "target": 1, "value": 5}]}
    chord_diagram    │ {"matrix": [[0,5,3],[5,0,2],[3,2,0]], "labels": ["A","B","C"]}
    sankey_diagram   │ {"nodes": [{"name": "Solar", "x": 0}, {"name": "Grid", "x": 1}],
                     │  "links": [{"source": 0, "target": 1, "value": 40}]}

    ═══════════════════════════════════════════════════════════════════════════
    HIGHCHARTS CHARTS (3 types — interactive time series & calendars):
    ═══════════════════════════════════════════════════════════════════════════
    streamgraphs     │ {"dates": ["2024-01","2024-02",...],
                     │  "series_dict": {"Category A": [10,20,...], "Category B": [5,15,...]},
                     │  "annotation": {"Launch event": 3}, "area": true}
                     │  annotation = {label: index_in_dates}. area=true for area chart, false for streamgraph.
    calendar_heatmap │ {"series_dict": {"2024-01-01": 0.85, "2024-01-02": 0.72, ...},
                     │  "min": 0, "max": 1}
                     │  Values are displayed as colors. min/max define the color scale range.
    calendar_gantt   │ {"series_dict": [<highcharts gantt series objects>],
                     │  "min_date": "2024-01-01", "max_date": "2024-12-31"}

    ═══════════════════════════════════════════════════════════════════════════
    CHART SELECTION TIPS:
    ═══════════════════════════════════════════════════════════════════════════
    • Distribution of one variable       → plotly.histogram or rawgraphs.violinplot
    • Compare categories                 → rawgraphs.barchart or rawgraphs.radarchart
    • Trends over time                   → plotly.linechart or highcharts.streamgraphs
    • Part-to-whole                      → rawgraphs.piechart, sunburst, or treemap
    • Relationships / correlations       → rawgraphs.bubblechart or parallelcoordinates
    • Flows / journeys / conversions     → rawgraphs.sankey_diagram or alluvial_diagram
    • Hierarchical data                  → rawgraphs.treemap, sunburst, or circlepacking
    • Calendar patterns / daily metrics  → highcharts.calendar_heatmap
    • Project timelines                  → rawgraphs.gantt_chart or highcharts.calendar_gantt
    • Ranking changes over time          → rawgraphs.bumpchart or slopechart
    • Correlation matrix                 → rawgraphs.matrixplot
    • Multi-dimensional comparison       → rawgraphs.parallelcoordinates or radarchart
    """
    report = _get_report(report_id)
    if not report:
        return json.dumps({"error": f"Report '{report_id}' not found. Use report_create first."})

    chart_data = json.loads(data) if isinstance(data, str) else data

    try:
        if engine == "plotly":
            from scomp_link.utils import plotly_utils

            _PLOTLY_TYPES = {"histogram", "barchart", "linechart", "area_chart"}
            if chart_type not in _PLOTLY_TYPES:
                return json.dumps(
                    {"error": f"Unknown plotly chart_type '{chart_type}'. Valid: {sorted(_PLOTLY_TYPES)}"}
                )
            if chart_type == "histogram":
                fig = plotly_utils.histogram(chart_data["values"], chart_data.get("name", title))
            elif chart_type == "barchart":
                fig = plotly_utils.barchart(
                    chart_data["categories"], chart_data["values"], y_axis_titles=chart_data.get("y_axis_titles")
                )
            elif chart_type == "linechart":
                fig = plotly_utils.linechart(
                    chart_data["dates"],
                    chart_data["lines"],
                    title_text=title,
                    y_labels=chart_data.get("y_labels", "value"),
                    format_date=chart_data.get("format_date", "%Y-%m-%d"),
                )
            elif chart_type == "area_chart":
                fig = plotly_utils.area_chart(
                    chart_data["dates"],
                    chart_data["lines"],
                    title_text=title,
                    y_labels=chart_data.get("y_labels", "value"),
                    format_date=chart_data.get("format_date", "%Y-%m-%d"),
                )
            report.add_graph_to_report(fig, title)

        elif engine == "rawgraphs":
            from scomp_link.utils import rawgraphs

            func = getattr(rawgraphs, chart_type, None)
            if func is None:
                return json.dumps({"error": f"Unknown rawgraphs chart_type '{chart_type}'. Valid: {rawgraphs.__all__}"})
            # Pass title and chart_data as kwargs
            kwargs = dict(chart_data)
            kwargs.setdefault("title", title)
            svg = func(**kwargs)
            report.add_rawgraphs_to_report(svg, title)

        elif engine == "highcharts":
            from scomp_link.utils import highcharts

            _HC_TYPES = {"streamgraphs", "calendar_heatmap", "calendar_gantt"}
            if chart_type not in _HC_TYPES:
                return json.dumps(
                    {"error": f"Unknown highcharts chart_type '{chart_type}'. Valid: {sorted(_HC_TYPES)}"}
                )
            func = getattr(highcharts, chart_type)
            kwargs = dict(chart_data)
            html = func(title, **kwargs)
            report.html_report += html

        else:
            return json.dumps({"error": f"Unknown engine '{engine}'. Valid: plotly, rawgraphs, highcharts"})

    except Exception as e:
        return json.dumps({"error": f"Chart generation failed: {type(e).__name__}: {str(e)}"})

    return json.dumps({"status": "chart_added", "engine": engine, "chart_type": chart_type, "title": title})


@mcp.tool()
def report_save(report_id: str, output: str = "report.html") -> str:
    """Save the report to an HTML file and close the session.

    Parameters:
        report_id: The report ID returned by report_create.
        output: Output file path (e.g. "my_report.html").

    After saving, the report session is removed from memory. To create another report,
    call report_create again.
    """
    report = _get_report(report_id)
    if not report:
        return json.dumps({"error": f"Report '{report_id}' not found. Use report_create first."})

    # Close any open section
    if report.section_just_open:
        report.close_section()

    report.save_html(output)
    del _reports[report_id]

    size_kb = os.path.getsize(output) / 1024
    return json.dumps({"status": "saved", "path": output, "size_kb": round(size_kb, 1)})


# ═══════════════════════════════════════════════════════════════════
# RESOURCES
# ═══════════════════════════════════════════════════════════════════


@mcp.resource("scomp://artifact/{path}")
def get_artifact_info(path: str) -> str:
    """Inspect a .scomp artifact: model type, config, metrics, feature schema."""
    import scomp_link

    scomp_link.set_verbosity("silent")
    if not os.path.exists(path):
        return json.dumps({"error": f"Artifact not found: {path}"})
    loaded = scomp_link.ScompArtifact.load(path)
    return json.dumps(loaded.info(), indent=2, default=str)


@mcp.resource("scomp://data/{path}")
def get_data_info(path: str) -> str:
    """Get schema and sample rows from a dataset file."""
    import pandas as pd

    if not os.path.exists(path):
        return json.dumps({"error": f"File not found: {path}"})
    df = _load_df(path)
    schema = {col: str(df[col].dtype) for col in df.columns}
    sample = df.head(5).to_dict("records")
    return json.dumps(
        {"path": path, "shape": list(df.shape), "schema": schema, "sample": sample}, indent=2, default=str
    )


@mcp.resource("scomp://models")
def get_available_models() -> str:
    """List all model types available in scomp-link's model factory."""
    models = {
        "regression": [
            "Econometric Model",
            "Ridge / SVR",
            "Lasso / Elastic Net",
            "Gradient Boosting / Random Forest",
            "SGD Regressor",
        ],
        "classification": [
            "Naive Bayes / Classification Tree",
            "SVC / K-Neighbors",
            "SGD / Gradient Boosting / Random Forest",
            "CNN (images)",
        ],
        "clustering": ["KMeans / Hierarchical", "Mean-Shift"],
        "text": ["TF-IDF + SGD", "Contrastive (BERT)"],
        "anomaly": ["Isolation Forest", "LOF", "TabNet", "Transformer"],
        "time_series": ["ARIMA", "Exponential Smoothing", "Auto"],
    }
    return json.dumps(models, indent=2)


# ═══════════════════════════════════════════════════════════════════
# PROMPTS
# ═══════════════════════════════════════════════════════════════════


@mcp.prompt()
def ml_workflow(data_path: str, target: str, task: str = "regression") -> str:
    """Template for a complete ML workflow from data to deployed model."""
    return f"""You are building an ML pipeline with scomp-link.

Dataset: {data_path}
Target column: {target}
Task: {task}

Follow these steps:
1. Profile the data: use describe_data("{data_path}")
2. Check quality issues (missing values, outliers, correlations)
3. Engineer features if needed: use engineer_features("{data_path}", "{target}")
4. Train and tune: use train_model with tune=true
5. Validate: use validate_model on held-out test data
6. If metrics are good, save the artifact and optionally serve it

Start by profiling the data to understand its structure."""


@mcp.prompt()
def debug_model(artifact_path: str, issue: str = "low accuracy") -> str:
    """Template for diagnosing and fixing model performance issues."""
    return f"""You are debugging a model with poor performance.

Artifact: {artifact_path}
Issue: {issue}

Diagnostic steps:
1. Inspect the artifact: read resource scomp://artifact/{artifact_path}
2. Check for data drift: use detect_drift between training reference and current data
3. Look at feature importance: validate the model and check which features matter
4. Check for fairness issues that might mask real problems
5. Try feature engineering: interactions and log transforms often help
6. Consider tuning: more n_trials or different model architecture

Common fixes:
- Low R²/accuracy → more features, tune hyperparameters, check data quality
- High variance → reduce model complexity, add regularization
- Drift detected → retrain on recent data
- Fairness issues → examine sensitive features, consider rebalancing"""


@mcp.prompt()
def monitor_production(reference: str, current: str, artifact: Optional[str] = None) -> str:
    """Template for production data monitoring workflow."""
    return f"""You are monitoring production data quality.

Reference (training) data: {reference}
Current (production) data: {current}
Model artifact: {artifact or 'not provided'}

Steps:
1. Check for drift: use detect_drift("{reference}", "{current}")
2. Scan for anomalies: use detect_anomalies on current data
3. If artifact provided, validate current performance: use validate_model
4. Generate a monitoring report: use generate_report with both data paths
5. Flag any features with PSI > 0.2 as drifted — may need retraining"""


@mcp.prompt()
def create_dashboard(data_path: str, title: str = "Dashboard") -> str:
    """Template for creating an HTML analytical dashboard with charts."""
    return f"""You are creating an interactive HTML dashboard.

Data: {data_path}
Title: {title}

Steps:
1. First, profile the data: use describe_data("{data_path}")
2. Plan sections based on the data types found:
   - Overview: key statistics table
   - Distributions: histograms for numeric columns
   - Trends: linechart or area_chart if time-based data exists
   - Comparisons: barchart for categorical breakdowns
   - Correlations: heatmap or parallel coordinates for multi-dimensional
   - Hierarchies: treemap or sunburst if hierarchical structure
   - Flows: sankey if flow/journey data
3. Generate the report: use generate_report("{data_path}", output="{title.lower().replace(' ', '_')}.html", title="{title}")
4. For custom charts, use create_visualization with specific chart types

Available chart engines:
- Plotly (interactive): histogram, barchart, linechart, area_chart
- RAWGraphs (SVG, publication-quality): treemap, sankey, sunburst, chord, alluvial, dendrogram, etc.
- Highcharts (interactive time series): streamgraphs, calendar_heatmap, calendar_gantt"""


@mcp.prompt()
def build_custom_report(data_path: str, title: str = "Report") -> str:
    """Template for building a fully custom branded report step-by-step using report_* tools."""
    return f"""You are building a custom HTML report using scomp-link's report builder tools.

Data: {data_path}
Title: {title}

WORKFLOW:
1. Create the report:       report_create("{title}")
   → Returns a report_id. All subsequent calls use this ID.
   → Branding defaults come from config files (~/.scomp-link/config.yaml or .scomp-link.yaml).
   → Override any param directly: main_color="#CC0000", footer_html="<footer>...</footer>"

2. Structure with sections: report_add_section(report_id, "Section Title")
   → Sections are collapsible in the HTML output.
   → Each new section auto-closes the previous one.

3. Add content:
   → report_add_text(report_id, "Explanation...", style="paragraph")
   → report_add_table(report_id, '[{{"col": "val"}}]', "Table Title")
   → report_add_chart(report_id, "plotly", "linechart", data_json, "Trend")

4. Save:                    report_save(report_id, "{title.lower().replace(' ', '_')}.html")
   → Closes open sections, writes file, frees memory.

CHART ENGINE GUIDE:
• plotly     — Interactive (zoom, hover). 4 types: histogram, barchart, linechart, area_chart.
• rawgraphs  — Publication SVG. 31 types: treemap, sankey, sunburst, chord, violin, radar, etc.
• highcharts — Time series. 3 types: streamgraphs, calendar_heatmap, calendar_gantt.

RECOMMENDED REPORT STRUCTURE:
1. "Executive Summary" section with key metrics table
2. "Distributions" section with histograms for numeric columns
3. "Trends" section with linechart/area_chart for time-based data
4. "Comparisons" section with barchart/radarchart for categories
5. "Deep Dive" section with advanced charts (treemap, sankey, etc.)

Start by profiling the data with describe_data("{data_path}") to understand what charts make sense."""


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════


def _load_df(path: str):
    """Load a DataFrame from CSV, TSV, or Parquet."""
    from pathlib import Path

    import pandas as pd

    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    elif p.suffix in (".csv", ".tsv"):
        sep = "\t" if p.suffix == ".tsv" else ","
        return pd.read_csv(p, sep=sep)
    else:
        return pd.read_csv(p)


# ═══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════


def main():
    """Start the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
