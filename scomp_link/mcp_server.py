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

from mcp.server.fastmcp import FastMCP
from typing import Optional
import json
import os

mcp = FastMCP(
    "scomp-link",
    instructions="End-to-end ML toolkit: train models, tune hyperparameters, detect drift, "
                 "generate HTML reports with 39 chart types, detect anomalies, forecast time series, "
                 "check fairness, and serve models as REST APIs."
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
        row = {"column": col, "dtype": str(df[col].dtype),
               "missing_pct": round(df[col].isnull().mean() * 100, 1),
               "unique": int(df[col].nunique())}
        if pd.api.types.is_numeric_dtype(df[col]):
            row.update({"min": round(float(df[col].min()), 4), "max": round(float(df[col].max()), 4),
                        "mean": round(float(df[col].mean()), 4), "std": round(float(df[col].std()), 4)})
        rows.append(row)
    return json.dumps({"shape": list(df.shape), "columns": rows}, indent=2)


@mcp.tool()
def train_model(data: str, target: str, task: str = "regression",
                engineer: bool = False, tune: bool = False, n_trials: int = 50,
                save_artifact: Optional[str] = None) -> str:
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
        from scomp_link.models.advanced_tuning import OptunaOptimizer
        from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
        from sklearn.model_selection import train_test_split

        X = df.drop(columns=[target])
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        base = GradientBoostingRegressor if task == "regression" else GradientBoostingClassifier
        scoring = "r2" if task == "regression" else "accuracy"

        def param_space(trial):
            return {"n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 3, 12),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True)}

        opt = OptunaOptimizer(base, param_space, scoring=scoring, n_trials=n_trials)
        best_model = opt.optimize(X_train, y_train)

        y_pred = best_model.predict(X_test)
        if task == "regression":
            from sklearn.metrics import r2_score, mean_squared_error
            metrics = {"r2": round(r2_score(y_test, y_pred), 4),
                       "rmse": round(mean_squared_error(y_test, y_pred) ** 0.5, 4)}
        else:
            from sklearn.metrics import accuracy_score, f1_score
            metrics = {"accuracy": round(accuracy_score(y_test, y_pred), 4),
                       "f1": round(f1_score(y_test, y_pred, average="weighted", zero_division=0), 4)}

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
        result = {"status": results.get("status", "success"),
                  "model_type": results.get("model_type"),
                  "metrics": results.get("metrics")}
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
    import scomp_link
    import pandas as pd
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
    return json.dumps({"n_predictions": len(preds_list), "predictions": preds_list[:20],
                       "output_path": output}, default=str)


@mcp.tool()
def validate_model(artifact: str, data: str, target: str,
                   report: Optional[str] = None) -> str:
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

    return json.dumps({"task_type": task_type, "metrics": metrics, "n_samples": len(y),
                       "report_path": report}, indent=2, default=str)


@mcp.tool()
def detect_drift(reference: str, current: str, threshold: float = 0.2,
                 plot: Optional[str] = None) -> str:
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

    return json.dumps({"drifted_features": summary["drifted_features"],
                       "total_features": summary["total_features"],
                       "worst_feature": summary.get("worst_feature"),
                       "max_psi": round(summary.get("max_psi", 0), 4),
                       "plot_path": plot}, indent=2, default=str)


@mcp.tool()
def detect_anomalies(data: str, methods: str = "iforest,lof",
                     contamination: float = 0.05, consensus: int = 2) -> str:
    """Detect anomalies using multi-method consensus (Isolation Forest, LOF, TabNet, Transformer).
    Returns number of anomalies and per-method comparison."""
    import scomp_link
    scomp_link.set_verbosity("silent")

    df = _load_df(data)
    features = df.select_dtypes(include=["number"]).columns.tolist()
    method_list = [m.strip() for m in methods.split(",")]

    detector = scomp_link.AnomalyDetector(contamination=contamination, methods=method_list,
                                           consensus_threshold=consensus, verbose=False)
    results = detector.fit_predict(df, features=features)

    comparison = results["comparison"].to_dict("records")
    n_anomalies = int(results["data"]["is_anomaly"].sum())

    return json.dumps({"n_anomalies": n_anomalies, "total_rows": len(df),
                       "methods": comparison}, indent=2, default=str)


@mcp.tool()
def check_fairness(data: str, target: str, predicted: str, sensitive: str) -> str:
    """Compute fairness metrics: demographic parity, disparate impact (4/5 rule), equalized odds.
    Returns whether the model is fair and detailed group-level metrics."""
    import scomp_link
    scomp_link.set_verbosity("silent")

    df = _load_df(data)
    fm = scomp_link.FairnessMetrics(df[target].values, df[predicted].values,
                                     sensitive_feature=df[sensitive].values)
    report = fm.compute_all()

    return json.dumps({
        "demographic_parity": report["demographic_parity"],
        "disparate_impact": report["disparate_impact"],
        "equalized_odds": {"tpr_diff": report["equalized_odds"]["tpr_diff"],
                           "fpr_diff": report["equalized_odds"]["fpr_diff"],
                           "fair": report["equalized_odds"]["fair"]},
    }, indent=2, default=str)


@mcp.tool()
def forecast_series(data: str, column: str, horizon: int = 10,
                    method: str = "auto", plot: Optional[str] = None) -> str:
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
        fig.add_trace(go.Scatter(x=list(range(len(series), len(series) + horizon)),
                                  y=ci["forecast"].values, name="Forecast", line=dict(dash="dash")))
        fig.update_layout(title=f"Forecast: {column}")
        fig.write_html(plot)

    return json.dumps({"horizon": horizon, "method": method,
                       "forecast": ci["forecast"].round(4).tolist(),
                       "plot_path": plot}, indent=2, default=str)


@mcp.tool()
def engineer_features(data: str, target: str, interactions: bool = True,
                      log_transform: bool = True, output: Optional[str] = None) -> str:
    """Apply automated feature engineering: polynomial interactions, log transforms for skewed features,
    date extraction, target encoding. Returns output path and new shape."""
    import scomp_link
    import pandas as pd
    scomp_link.set_verbosity("silent")

    df = _load_df(data)
    y = df[target]
    X = df.drop(columns=[target])

    fe = scomp_link.FeatureEngineer(interactions=interactions, log_transform=log_transform)
    X_eng = fe.fit_transform(X, y)
    X_eng[target] = y.values

    out_path = output or data.replace(".csv", "_engineered.csv")
    X_eng.to_csv(out_path, index=False)

    return json.dumps({"output_path": out_path, "original_shape": list(df.shape),
                       "engineered_shape": list(X_eng.shape),
                       "new_columns": [c for c in X_eng.columns if c not in df.columns]}, indent=2)


@mcp.tool()
def cluster_data(data: str, n_clusters: int = 5, method: str = "kmeans",
                 features: Optional[str] = None, output: Optional[str] = None) -> str:
    """Cluster data using KMeans or MeanShift. Returns cluster labels and silhouette score.
    Optionally saves result with cluster column."""
    import scomp_link
    import pandas as pd
    import numpy as np
    from sklearn.metrics import silhouette_score
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

    return json.dumps({"n_clusters": int(len(np.unique(labels))),
                       "silhouette_score": round(sil, 4),
                       "cluster_sizes": {str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
                       "output_path": output}, indent=2)


@mcp.tool()
def generate_report(data: str, output: str = "report.html", title: str = "Analysis Report",
                    artifact: Optional[str] = None) -> str:
    """Generate an interactive HTML report. If only data is provided, creates an EDA report.
    If artifact is also provided, creates a model evaluation report with predictions and SHAP."""
    import scomp_link
    import pandas as pd
    scomp_link.set_verbosity("silent")

    from scomp_link.utils.report_html import ScompLinkHTMLReport
    from scomp_link.utils.plotly_utils import histogram, barchart

    df = _load_df(data)
    report = ScompLinkHTMLReport(title=title)

    # Overview section
    report.open_section("Dataset Overview")
    overview = pd.DataFrame([{"rows": len(df), "columns": len(df.columns),
                              "numeric": len(df.select_dtypes(include=["number"]).columns),
                              "categorical": len(df.select_dtypes(include=["object"]).columns),
                              "missing_total": int(df.isnull().sum().sum())}])
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
                df[target_col], preds, task_type=loaded.config.get("task_type", "regression"))
            report.add_dataframe(pd.DataFrame([metrics]), "Metrics")
            report.close_section()

    report.save_html(output)
    size_kb = os.path.getsize(output) / 1024
    return json.dumps({"report_path": output, "size_kb": round(size_kb, 1)}, indent=2)


@mcp.tool()
def create_visualization(chart_type: str, data: str, title: str = "Chart",
                         output: Optional[str] = None, **kwargs) -> str:
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
                f.write(f"<html><head><script src='https://code.highcharts.com/highcharts.js'></script></head><body>{html}</body></html>")
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
    import scomp_link
    import pandas as pd
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
    import scomp_link
    from pathlib import Path
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
def embed_text(artifact: str, data: str, text_col: str = "text",
               output: Optional[str] = None) -> str:
    """Generate embeddings from a trained contrastive text model (.scomp artifact).
    Returns embedding shape and optionally saves to .npy file.
    Use this to extract text representations for downstream analysis."""
    import scomp_link
    import numpy as np
    scomp_link.set_verbosity("silent")

    loaded = scomp_link.ScompArtifact.load(artifact)
    if not hasattr(loaded.model, 'embed'):
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
def select_backbone(data: str, text_col: str = "text", label_col: str = "label",
                    candidates: Optional[str] = None, sample_size: int = 500) -> str:
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
    results = selector.find_best_backbone(df, text_col=text_col, label_col=label_col,
                                           sample_size=sample_size)

    return json.dumps({
        "best_model": results.iloc[0]["model"],
        "best_loss": float(results.iloc[0]["loss"]),
        "ranking": results.to_dict("records"),
    }, indent=2, default=str)


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
    return json.dumps({"path": path, "shape": list(df.shape), "schema": schema,
                       "sample": sample}, indent=2, default=str)


@mcp.resource("scomp://models")
def get_available_models() -> str:
    """List all model types available in scomp-link's model factory."""
    models = {
        "regression": ["Econometric Model", "Ridge / SVR", "Lasso / Elastic Net",
                       "Gradient Boosting / Random Forest", "SGD Regressor"],
        "classification": ["Naive Bayes / Classification Tree", "SVC / K-Neighbors",
                           "SGD / Gradient Boosting / Random Forest", "CNN (images)"],
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


# ═══════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════

def _load_df(path: str):
    """Load a DataFrame from CSV, TSV, or Parquet."""
    import pandas as pd
    from pathlib import Path
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
