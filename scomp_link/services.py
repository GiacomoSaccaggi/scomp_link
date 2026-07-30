# -*- coding: utf-8 -*-
"""
███████╗███████╗██████╗ ██╗   ██╗██╗ ██████╗███████╗███████╗
██╔════╝██╔════╝██╔══██╗██║   ██║██║██╔════╝██╔════╝██╔════╝
███████╗█████╗  ██████╔╝██║   ██║██║██║     █████╗  ███████╗
╚════██║██╔══╝  ██╔══██╗╚██╗ ██╔╝██║██║     ██╔══╝  ╚════██║
███████║███████╗██║  ██║ ╚████╔╝ ██║╚██████╗███████╗███████║
╚══════╝╚══════╝╚═╝  ╚═╝  ╚═══╝  ╚═╝ ╚═════╝╚══════╝╚══════╝

Shared service layer — business logic used by both cli.py and mcp_server.py.

Neither CLI-specific code (argparse, sys.exit) nor MCP-specific code (FastMCP,
JSON serialisation) belongs here. Each function:
  1. Accepts a typed config object from schemas.py
  2. Executes the ML logic with lazy imports
  3. Returns a plain dict
  4. Raises a typed exception from exceptions.py on failure

Usage:
    from scomp_link.services import describe, train, load_dataframe
    from scomp_link.schemas import DescribeConfig, TrainConfig
    from scomp_link.exceptions import DataLoadError

    df = load_dataframe("data.csv")                         # DataLoadError on bad path
    result = describe(DescribeConfig(data="data.csv"))      # {shape, columns}
    result = train(TrainConfig(data="data.csv", target="y"))
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from scomp_link.exceptions import (
    ArtifactError,
    DataLoadError,
    DataValidationError,
    DriftDetectionError,
    ModelTrainingError,
)

if TYPE_CHECKING:
    import pandas as pd

    from scomp_link.schemas import (
        AnomalyConfig,
        ClusterConfig,
        DescribeConfig,
        DriftConfig,
        EngineerConfig,
        FairnessConfig,
        ForecastConfig,
        PredictConfig,
        TrainConfig,
        ValidateConfig,
    )


# ── Data loading ──────────────────────────────────────────────────────────────


def load_dataframe(path: str) -> "pd.DataFrame":
    """Load a CSV or Parquet file into a DataFrame.

    Raises:
        DataLoadError: If the file is not found, the format is unsupported,
            or the file cannot be parsed.
    """
    import pandas as pd

    p = Path(path)
    if not p.exists():
        raise DataLoadError(f"File not found: {path}")
    if p.suffix == ".parquet":
        try:
            return pd.read_parquet(p)
        except Exception as e:
            raise DataLoadError(f"Cannot read parquet file {path}: {e}") from e
    elif p.suffix in (".csv", ".tsv"):
        sep = "\t" if p.suffix == ".tsv" else ","
        try:
            return pd.read_csv(p, sep=sep)
        except Exception as e:
            raise DataLoadError(f"Cannot read CSV file {path}: {e}") from e
    else:
        raise DataLoadError(f"Unsupported file format '{p.suffix}' for {path}. Supported: .csv, .tsv, .parquet")


# ── Describe ──────────────────────────────────────────────────────────────────


def describe(config: "DescribeConfig") -> dict:
    """Profile a dataset column by column.

    Returns:
        {
            "shape": [rows, cols],
            "columns": [
                {"column": str, "dtype": str, "missing_pct": float,
                 "unique": int, "min"?: float, "max"?: float,
                 "mean"?: float, "std"?: float}
            ]
        }

    Raises:
        DataLoadError: If the dataset file cannot be read.
    """
    import pandas as pd

    df = load_dataframe(config.data)
    rows = []
    for col in df.columns:
        row: dict = {
            "column": col,
            "dtype": str(df[col].dtype),
            "missing_pct": round(float(df[col].isnull().mean() * 100), 1),
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
    return {"shape": list(df.shape), "columns": rows}


# ── Train ─────────────────────────────────────────────────────────────────────


def train(config: "TrainConfig") -> dict:
    """Train an ML model (optionally with feature engineering and/or Optuna tuning).

    Returns:
        {"status": "success", "model_type": str, "metrics": dict,
         "artifact_path"?: str}

    Raises:
        DataLoadError: If the dataset cannot be read.
        DataValidationError: If the target column is missing.
        ModelTrainingError: If training fails.
    """
    import scomp_link

    scomp_link.set_verbosity("silent")

    df = load_dataframe(config.data)
    if config.target not in df.columns:
        raise DataValidationError(f"Target column '{config.target}' not found. Available: {list(df.columns)}")

    try:
        if config.engineer:
            fe = scomp_link.FeatureEngineer(interactions=True, log_transform=True)
            y = df[config.target]
            X = fe.fit_transform(df.drop(columns=[config.target]), y)
            df = X.copy()
            df[config.target] = y.values

        if config.tune:
            from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
            from sklearn.model_selection import train_test_split

            from scomp_link.models.advanced_tuning import OptunaOptimizer

            X = df.drop(columns=[config.target])
            y = df[config.target]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            is_regression = config.task == "regression"
            base = GradientBoostingRegressor if is_regression else GradientBoostingClassifier
            scoring = "r2" if is_regression else "accuracy"

            def param_space(trial):
                return {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 3, 12),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                }

            opt = OptunaOptimizer(base, param_space, scoring=scoring, n_trials=config.n_trials)
            best_model = opt.optimize(X_train, y_train)
            y_pred = best_model.predict(X_test)

            if is_regression:
                from sklearn.metrics import mean_squared_error, r2_score

                metrics = {
                    "r2": round(float(r2_score(y_test, y_pred)), 4),
                    "rmse": round(float(mean_squared_error(y_test, y_pred) ** 0.5), 4),
                }
            else:
                from sklearn.metrics import accuracy_score, f1_score

                metrics = {
                    "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
                    "f1": round(float(f1_score(y_test, y_pred, average="weighted", zero_division=0)), 4),
                }

            result: dict = {
                "status": "success",
                "model_type": type(best_model).__name__,
                "metrics": metrics,
            }
            if config.save_artifact:
                artifact = scomp_link.ScompArtifact()
                artifact.set_model(best_model)
                artifact.set_config(task_type=config.task, target_col=config.target)
                artifact.set_metrics(metrics)
                artifact.set_feature_schema(X_train)
                artifact.save(config.save_artifact)
                result["artifact_path"] = config.save_artifact

        else:
            pipe = scomp_link.ScompLinkPipeline("services_train")
            pipe.import_and_clean_data(df)
            pipe.select_variables(target_col=config.target)
            objective = "numerical_prediction" if config.task == "regression" else "categorical_known"
            pipe.choose_model(objective)
            results = pipe.run_pipeline(task_type=config.task)
            result = {
                "status": results.get("status", "success"),
                "model_type": results.get("model_type"),
                "metrics": results.get("metrics"),
            }
            if config.save_artifact:
                artifact = scomp_link.ScompArtifact()
                artifact.set_model(pipe.model)
                artifact.set_config(task_type=config.task, target_col=config.target)
                artifact.set_metrics(results.get("metrics", {}))
                artifact.save(config.save_artifact)
                result["artifact_path"] = config.save_artifact

    except (DataValidationError, DataLoadError):
        raise
    except Exception as e:
        raise ModelTrainingError(f"Training failed: {e}") from e

    return result


# ── Predict ───────────────────────────────────────────────────────────────────


def predict_from_artifact(config: "PredictConfig") -> dict:
    """Generate predictions from a saved .scomp artifact.

    Returns:
        {"n_predictions": int, "predictions": list, "output_path"?: str}

    Raises:
        ArtifactError: If the artifact file cannot be loaded.
        DataLoadError: If the input data file cannot be read.
    """
    import scomp_link

    scomp_link.set_verbosity("silent")

    if not Path(config.artifact).exists():
        raise ArtifactError(f"Artifact not found: {config.artifact}")
    try:
        loaded = scomp_link.ScompArtifact.load(config.artifact)
    except Exception as e:
        raise ArtifactError(f"Cannot load artifact {config.artifact}: {e}") from e

    df = load_dataframe(config.data)
    target_col = loaded.config.get("target_col")
    feature_cols = [c for c in df.columns if c != target_col]
    predictions = loaded.predict(df[feature_cols])

    if config.output:
        out_df = df.copy()
        out_df["prediction"] = predictions
        out_df.to_csv(config.output, index=False)

    preds_list = predictions.tolist() if hasattr(predictions, "tolist") else list(predictions)
    return {
        "n_predictions": len(preds_list),
        "predictions": preds_list[:20],
        "output_path": config.output,
    }


# ── Validate ──────────────────────────────────────────────────────────────────


def validate(config: "ValidateConfig") -> dict:
    """Evaluate a .scomp artifact on labelled test data.

    Returns:
        {"task_type": str, "metrics": dict, "n_samples": int, "report_path"?: str}

    Raises:
        ArtifactError: If the artifact cannot be loaded.
        DataLoadError: If the test data cannot be read.
        DataValidationError: If the target column is missing.
    """
    import scomp_link

    scomp_link.set_verbosity("silent")

    if not Path(config.artifact).exists():
        raise ArtifactError(f"Artifact not found: {config.artifact}")
    try:
        loaded = scomp_link.ScompArtifact.load(config.artifact)
    except Exception as e:
        raise ArtifactError(f"Cannot load artifact {config.artifact}: {e}") from e

    df = load_dataframe(config.data)
    if config.target not in df.columns:
        raise DataValidationError(f"Target column '{config.target}' not found. Available: {list(df.columns)}")

    feature_cols = [c for c in df.columns if c != config.target]
    X, y = df[feature_cols], df[config.target]
    predictions = loaded.predict(X)

    task_type = loaded.config.get("task_type", "regression")
    validator = scomp_link.Validator(loaded.model)
    metrics = validator.evaluate(y, predictions, task_type=task_type)

    if config.report:
        validator.generate_validation_report(y, predictions, task_type=task_type, report_name=config.report)

    return {
        "task_type": task_type,
        "metrics": metrics,
        "n_samples": int(len(y)),
        "report_path": config.report,
    }


# ── Drift Detection ───────────────────────────────────────────────────────────


def detect_drift(config: "DriftConfig") -> dict:
    """Detect distribution drift between reference and current datasets.

    Returns:
        {"drifted_features": int, "total_features": int,
         "worst_feature": str, "max_psi": float, "plot_path"?: str}

    Raises:
        DataLoadError: If either dataset cannot be read.
        DriftDetectionError: If no common numeric columns exist.
    """
    import scomp_link

    scomp_link.set_verbosity("silent")

    df_ref = load_dataframe(config.reference)
    df_cur = load_dataframe(config.current)

    numeric_cols = df_ref.select_dtypes(include=["number"]).columns.tolist()
    common = [c for c in numeric_cols if c in df_cur.columns]
    if not common:
        raise DriftDetectionError(
            "No common numeric columns between reference and current datasets. "
            f"Reference numeric columns: {numeric_cols}"
        )

    detector = scomp_link.DriftDetector(df_ref[common], psi_threshold=config.threshold)
    report = detector.detect(df_cur[common])
    summary = detector.summary(report)

    if config.plot:
        fig = detector.plot_drift_report(report)
        fig.write_html(config.plot)

    return {
        "drifted_features": summary["drifted_features"],
        "total_features": summary["total_features"],
        "worst_feature": summary.get("worst_feature"),
        "max_psi": round(float(summary.get("max_psi", 0)), 4),
        "plot_path": config.plot,
    }


# ── Anomaly Detection ─────────────────────────────────────────────────────────


def detect_anomalies(config: "AnomalyConfig") -> dict:
    """Detect anomalies using multi-method consensus.

    Returns:
        {"n_anomalies": int, "total_rows": int, "methods": list[dict]}

    Raises:
        DataLoadError: If the dataset cannot be read.
    """
    import numpy as np

    import scomp_link

    scomp_link.set_verbosity("silent")

    df = load_dataframe(config.data)
    features = df.select_dtypes(include=["number"]).columns.tolist()
    method_list = [m.strip() for m in config.methods.split(",")]

    detector = scomp_link.AnomalyDetector(
        contamination=config.contamination,
        methods=method_list,
        consensus_threshold=config.consensus,
        verbose=False,
    )
    results = detector.fit_predict(df, features=features)
    comparison = results["comparison"].to_dict("records")
    n_anomalies = int(results["data"]["is_anomaly"].sum())

    return {
        "n_anomalies": n_anomalies,
        "total_rows": int(len(df)),
        "methods": comparison,
    }


# ── Fairness ──────────────────────────────────────────────────────────────────


def check_fairness(config: "FairnessConfig") -> dict:
    """Compute fairness metrics on model predictions.

    Returns:
        {"demographic_parity": dict, "disparate_impact": dict, "equalized_odds": dict}

    Raises:
        DataLoadError: If the dataset cannot be read.
        DataValidationError: If required columns are missing.
    """
    import scomp_link

    scomp_link.set_verbosity("silent")

    df = load_dataframe(config.data)
    for col in (config.target, config.predicted, config.sensitive):
        if col not in df.columns:
            raise DataValidationError(f"Column '{col}' not found. Available: {list(df.columns)}")

    fm = scomp_link.FairnessMetrics(
        df[config.target].values,
        df[config.predicted].values,
        sensitive_feature=df[config.sensitive].values,
    )
    report = fm.compute_all()

    return {
        "demographic_parity": report["demographic_parity"],
        "disparate_impact": report["disparate_impact"],
        "equalized_odds": {
            "tpr_diff": report["equalized_odds"]["tpr_diff"],
            "fpr_diff": report["equalized_odds"]["fpr_diff"],
            "fair": report["equalized_odds"]["fair"],
        },
    }


# ── Forecast ──────────────────────────────────────────────────────────────────


def forecast(config: "ForecastConfig") -> dict:
    """Forecast future values of a time series column.

    Returns:
        {"horizon": int, "method": str, "forecast": list[float], "plot_path"?: str}

    Raises:
        DataLoadError: If the dataset cannot be read.
        DataValidationError: If the column is missing or has too few values.
    """
    import scomp_link

    scomp_link.set_verbosity("silent")

    df = load_dataframe(config.data)
    if config.column not in df.columns:
        raise DataValidationError(f"Column '{config.column}' not found. Available: {list(df.columns)}")

    series = df[config.column].dropna()
    if len(series) < 10:
        raise DataValidationError(
            f"Column '{config.column}' has only {len(series)} non-null values. "
            "At least 10 observations are required for forecasting."
        )

    fc = scomp_link.TimeSeriesForecaster(method=config.method, horizon=config.horizon)
    fc.fit(series)
    ci = fc.predict_with_ci(steps=config.horizon)

    if config.plot:
        import plotly.graph_objects as go

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=series.values, name="Historical"))
        fig.add_trace(
            go.Scatter(
                x=list(range(len(series), len(series) + config.horizon)),
                y=ci["forecast"].values,
                name="Forecast",
                line={"dash": "dash"},
            )
        )
        fig.update_layout(title=f"Forecast: {config.column}")
        fig.write_html(config.plot)

    return {
        "horizon": config.horizon,
        "method": config.method,
        "forecast": ci["forecast"].round(4).tolist(),
        "plot_path": config.plot,
    }


# ── Feature Engineering ───────────────────────────────────────────────────────


def engineer(config: "EngineerConfig") -> dict:
    """Apply automated feature engineering to a dataset.

    Returns:
        {"output_path": str, "original_shape": list, "engineered_shape": list,
         "new_columns": list[str]}

    Raises:
        DataLoadError: If the dataset cannot be read.
        DataValidationError: If the target column is missing.
    """
    import scomp_link

    scomp_link.set_verbosity("silent")

    df = load_dataframe(config.data)
    if config.target not in df.columns:
        raise DataValidationError(f"Target column '{config.target}' not found. Available: {list(df.columns)}")

    y = df[config.target]
    X = df.drop(columns=[config.target])

    fe = scomp_link.FeatureEngineer(
        interactions=config.interactions,
        log_transform=config.log_transform,
    )
    X_eng = fe.fit_transform(X, y)
    X_eng[config.target] = y.values

    out_path = config.output or str(Path(config.data).with_stem(Path(config.data).stem + "_engineered"))
    X_eng.to_csv(out_path, index=False)

    return {
        "output_path": out_path,
        "original_shape": list(df.shape),
        "engineered_shape": list(X_eng.shape),
        "new_columns": [c for c in X_eng.columns if c not in df.columns],
    }


# ── Clustering ────────────────────────────────────────────────────────────────


def cluster(config: "ClusterConfig") -> dict:
    """Cluster a dataset and optionally save the labelled result.

    Returns:
        {"n_clusters": int, "silhouette_score": float,
         "cluster_sizes": dict[str, int], "output_path"?: str}

    Raises:
        DataLoadError: If the dataset cannot be read.
        DataValidationError: If no numeric columns are found.
    """
    import numpy as np
    from sklearn.metrics import silhouette_score

    df = load_dataframe(config.data)
    feat_cols = (
        [f.strip() for f in config.features.split(",")]
        if config.features
        else df.select_dtypes(include=["number"]).columns.tolist()
    )
    if not feat_cols:
        raise DataValidationError(f"No numeric columns found in {config.data}.")

    X = df[feat_cols].values

    if config.method == "kmeans":
        from sklearn.cluster import KMeans

        model = KMeans(n_clusters=config.n_clusters, random_state=42, n_init="auto")
    else:
        from sklearn.cluster import MeanShift

        model = MeanShift()

    labels = model.fit_predict(X)
    sil = float(silhouette_score(X, labels))

    if config.output:
        out_df = df.copy()
        out_df["cluster"] = labels
        out_df.to_csv(config.output, index=False)

    unique, counts = np.unique(labels, return_counts=True)
    return {
        "n_clusters": int(len(unique)),
        "silhouette_score": round(sil, 4),
        "cluster_sizes": {str(k): int(v) for k, v in zip(unique, counts)},
        "output_path": config.output,
    }
