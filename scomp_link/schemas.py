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


# ── LLM tool schemas ─────────────────────────────────────────────────────────


class LLMFineTuneConfig(BaseModel):
    """Configuration for the llm_finetune MCP tool."""

    model_config = _cfg

    model: str = Field(..., description="HuggingFace model identifier or local path.")
    method: Literal["lora", "qlora", "full"] = Field("lora", description="Fine-tuning method.")
    data: str = Field(..., description="Path to the training dataset.")
    epochs: int = Field(3, ge=1, description="Number of training epochs.")
    batch_size: int = Field(4, ge=1, description="Per-device batch size.")
    learning_rate: float = Field(2e-4, gt=0, description="Learning rate.")


class LLMConvertConfig(BaseModel):
    """Configuration for the llm_convert MCP tool."""

    model_config = _cfg

    model_path: str = Field(..., description="Path to the HuggingFace model directory.")
    quantization: str = Field("Q4_K_M", description="GGUF quantization level.")
    output_dir: Optional[str] = Field(None, description="Output directory for the GGUF file.")
    importance_matrix: Optional[str] = Field(None, description="Path to an importance matrix file.")


class LLMScratchConfig(BaseModel):
    """Configuration for the llm_scratch MCP tool."""

    model_config = _cfg

    vocab_size: int = Field(32_000, ge=1, description="Vocabulary size.")
    d_model: int = Field(768, ge=1, description="Model embedding dimension.")
    n_heads: int = Field(12, ge=1, description="Number of attention heads.")
    n_layers: int = Field(12, ge=1, description="Number of transformer layers.")
    data: str = Field(..., description="Path to the training corpus.")
    epochs: int = Field(10, ge=1, description="Number of training epochs.")


class LLMEstimateConfig(BaseModel):
    """Configuration for the llm_estimate MCP tool."""

    model_config = _cfg

    model_path: str = Field(..., description="Path to the HuggingFace model directory.")
    quantization: str = Field("Q4_K_M", description="GGUF quantization level to estimate.")


class LLMEvaluateConfig(BaseModel):
    """Configuration for the llm_evaluate MCP tool."""

    model_config = _cfg

    generated: str = Field(..., description="Path to generated text file, or the text itself.")
    references: Optional[str] = Field(None, description="Path to reference text file, or the text itself.")


class LLMFormatConfig(BaseModel):
    """Configuration for the llm_format MCP tool."""

    model_config = _cfg

    input_path: str = Field(..., description="Path to input dataset (JSON or JSONL).")
    output_path: str = Field(..., description="Path to output JSONL file.")
    source_format: str = Field(..., description="Source format: alpaca, sharegpt, or openai.")
    target_format: str = Field("chatml", description="Target format: chatml, llama, alpaca, or plain.")


class LLMDedupConfig(BaseModel):
    """Configuration for the llm_dedup MCP tool."""

    model_config = _cfg

    data: str = Field(..., description="Path to text file (one text per line) or JSON/JSONL.")
    method: str = Field("exact", description="Dedup method: exact or ngram.")
    threshold: float = Field(0.8, ge=0.0, le=1.0, description="Jaccard similarity threshold for ngram dedup.")


class LLMMergeConfig(BaseModel):
    """Configuration for the llm_merge MCP tool."""

    model_config = _cfg

    base_model: str = Field(..., description="Path to base model (shared pretrained ancestor).")
    models: str = Field(..., description="Comma-separated paths to fine-tuned models to merge.")
    method: str = Field("ties", description="Merge method: linear, slerp, ties, or dare.")
    density: float = Field(0.5, gt=0.0, le=1.0, description="Density for TIES/DARE (fraction to keep).")
    output_dir: str = Field("./merged", description="Output directory for merged model.")


class LLMServeConfig(BaseModel):
    """Configuration for the llm_serve MCP tool (info only — does not start the server)."""

    model_config = _cfg

    model_path: str = Field(..., description="Path to HuggingFace model directory.")
    port: int = Field(8080, ge=1, le=65535, description="Port for the inference server.")
    load_in_4bit: bool = Field(False, description="Load model in 4-bit quantization.")
