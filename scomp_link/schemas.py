# -*- coding: utf-8 -*-
"""
 ██████╗ ██████╗██╗  ██╗███████╗███╗   ███╗ █████╗ ███████╗
██╔════╝██╔════╝██║  ██║██╔════╝████╗ ████║██╔══██╗██╔════╝
╚█████╗ ██║     ███████║█████╗  ██╔████╔██║███████║███████╗
 ╚═══██╗██║     ██╔══██║██╔══╝  ██║╚██╔╝██║██╔══██║╚════██║
██████╔╝╚██████╗██║  ██║███████╗██║ ╚═╝ ██║██║  ██║███████║
╚═════╝  ╚═════╝╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝

Pydantic v2 input/output schemas for all scomp-link CLI commands and MCP tools.

Input models validate and coerce arguments before they reach services.py.
All heavy imports (pandas, sklearn, etc.) are kept out of this module so it
loads in milliseconds even without optional dependencies installed.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

# ── Shared config ─────────────────────────────────────────────────────────────

_cfg = ConfigDict(str_strip_whitespace=True, frozen=False)


# ── Input schemas ─────────────────────────────────────────────────────────────


class DescribeConfig(BaseModel):
    """Configuration for the describe (data profiling) tool."""

    model_config = _cfg

    data: str = Field(..., description="Path to the dataset file (CSV or Parquet).")


class TrainConfig(BaseModel):
    """Configuration for the train tool."""

    model_config = _cfg

    data: str = Field(..., description="Path to the training dataset.")
    target: str = Field(..., description="Name of the target column.")
    task: Literal["regression", "classification", "clustering"] = Field("regression", description="ML task type.")
    engineer: bool = Field(False, description="Apply automatic feature engineering before training.")
    tune: bool = Field(False, description="Run Optuna hyperparameter tuning instead of default training.")
    n_trials: int = Field(50, ge=1, le=500, description="Number of Optuna trials (only used when tune=True).")
    save_artifact: Optional[str] = Field(None, description="Path to save the .scomp artifact. Omit to skip saving.")


class PredictConfig(BaseModel):
    """Configuration for the predict tool."""

    model_config = _cfg

    artifact: str = Field(..., description="Path to the .scomp artifact.")
    data: str = Field(..., description="Path to the input dataset.")
    output: Optional[str] = Field(None, description="Path to save predictions CSV. Omit to return in-memory only.")


class ValidateConfig(BaseModel):
    """Configuration for the validate tool."""

    model_config = _cfg

    artifact: str = Field(..., description="Path to the .scomp artifact.")
    data: str = Field(..., description="Path to the test dataset.")
    target: str = Field(..., description="Name of the target column in the test dataset.")
    report: Optional[str] = Field(None, description="Path to save the HTML validation report.")


class DriftConfig(BaseModel):
    """Configuration for the detect_drift tool."""

    model_config = _cfg

    reference: str = Field(..., description="Path to the reference (training) dataset.")
    current: str = Field(..., description="Path to the current (production) dataset.")
    threshold: float = Field(
        0.2, gt=0.0, lt=10.0, description="PSI threshold above which a feature is considered drifted."
    )
    plot: Optional[str] = Field(None, description="Path to save the drift HTML plot.")


class AnomalyConfig(BaseModel):
    """Configuration for the detect_anomalies tool."""

    model_config = _cfg

    data: str = Field(..., description="Path to the dataset.")
    methods: str = Field(
        "iforest,lof", description="Comma-separated list of methods: iforest, lof, tabnet, transformer."
    )
    contamination: float = Field(0.05, gt=0.0, lt=1.0, description="Expected fraction of anomalies in the data.")
    consensus: int = Field(2, ge=1, description="Minimum number of methods that must agree to flag an anomaly.")


class FairnessConfig(BaseModel):
    """Configuration for the check_fairness tool."""

    model_config = _cfg

    data: str = Field(..., description="Path to the predictions dataset.")
    target: str = Field(..., description="Name of the ground-truth label column.")
    predicted: str = Field(..., description="Name of the model predictions column.")
    sensitive: str = Field(..., description="Name of the sensitive attribute column (e.g. gender, age_group).")


class ForecastConfig(BaseModel):
    """Configuration for the forecast tool."""

    model_config = _cfg

    data: str = Field(..., description="Path to the time series dataset.")
    column: str = Field(..., description="Name of the column to forecast.")
    horizon: int = Field(10, ge=1, le=1000, description="Number of future steps to forecast.")
    method: Literal["auto", "arima", "exp_smoothing"] = Field(
        "auto", description="Forecasting method. 'auto' selects the best available."
    )
    plot: Optional[str] = Field(None, description="Path to save the forecast HTML plot.")


class EngineerConfig(BaseModel):
    """Configuration for the engineer_features tool."""

    model_config = _cfg

    data: str = Field(..., description="Path to the input dataset.")
    target: str = Field(..., description="Name of the target column (used for target encoding).")
    interactions: bool = Field(True, description="Generate polynomial interaction features.")
    log_transform: bool = Field(True, description="Apply log1p transform to right-skewed numeric features.")
    output: Optional[str] = Field(
        None, description="Path to save the engineered dataset. Defaults to <data>_engineered.csv."
    )


class ClusterConfig(BaseModel):
    """Configuration for the cluster_data tool."""

    model_config = _cfg

    data: str = Field(..., description="Path to the dataset.")
    n_clusters: int = Field(5, ge=2, le=1000, description="Number of clusters (KMeans only).")
    method: Literal["kmeans", "meanshift"] = Field("kmeans", description="Clustering algorithm.")
    features: Optional[str] = Field(
        None, description="Comma-separated list of feature columns. Defaults to all numeric columns."
    )
    output: Optional[str] = Field(None, description="Path to save the dataset with cluster labels appended.")
