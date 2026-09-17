"""
LLM pipeline DSL steps for the >> operator.

Provides LLMStep (base), LLMFineTuneStep, LLMConvertStep, and LLMSaveStep
that compose into executable chains via the existing Chain infrastructure
in ``scomp_link.pipeline_dsl``.

Usage::

    from scomp_link.llm.dsl import LLMFineTuneStep, LLMConvertStep, LLMSaveStep

    chain = (
        LLMFineTuneStep("meta-llama/Llama-3-8B", method="lora", dataset="data.json", epochs=3)
        >> LLMConvertStep(quantization="Q4_K_M")
        >> LLMSaveStep("model.scomp")
    )
    chain.run()
"""

from __future__ import annotations

from abc import ABC
from pathlib import Path
from typing import Any, Literal

from scomp_link.pipeline_dsl import Step


class LLMStep(Step, ABC):
    """Marker base for steps that operate on LLM pipelines."""


class LLMFineTuneStep(LLMStep):
    """Fine-tune a pretrained model, returning a :class:`TrainResult`.

    Parameters
    ----------
    model : str
        HuggingFace model identifier or local path.
    method : str
        Fine-tuning method: ``"lora"``, ``"qlora"``, or ``"full"``.
    dataset : str
        Path to the training dataset (CSV, JSON, JSONL, Parquet).
    epochs : int
        Number of training epochs.
    batch_size : int
        Per-device training batch size.
    learning_rate : float
        Peak learning rate.
    config_overrides : dict | None
        Extra keyword arguments forwarded to :class:`FineTuneConfig`.
    """

    def __init__(
        self,
        model: str,
        method: Literal["lora", "qlora", "full"] = "lora",
        dataset: str = "data.json",
        epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        config_overrides: dict[str, Any] | None = None,
    ):
        self.model = model
        self.method: Literal["lora", "qlora", "full"] = method
        self.dataset = dataset
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.config_overrides = config_overrides or {}

    def execute(self, target: Any) -> Any:
        from scomp_link.llm.core.configs import FineTuneConfig
        from scomp_link.llm.training.finetune import FineTuner

        config = FineTuneConfig(method=self.method, **self.config_overrides)
        ft = FineTuner(self.model, method=self.method, config=config)
        return ft.train(
            self.dataset,
            epochs=self.epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
        )


class LLMConvertStep(LLMStep):
    """Convert a model to GGUF format, returning a :class:`ConvertResult`.

    Receives the model path from an upstream step's output (a
    :class:`TrainResult` whose ``model_path`` is used) or from the
    explicit ``model_path`` argument.

    Parameters
    ----------
    quantization : str
        Target GGUF quantization level (e.g. ``"Q4_K_M"``).
    model_path : str | Path | None
        Explicit model directory.  When *None*, the step extracts the
        path from the upstream ``TrainResult.model_path``.
    output_dir : str | Path | None
        Where to write the GGUF file.
    """

    def __init__(
        self,
        quantization: str = "Q4_K_M",
        model_path: str | Path | None = None,
        output_dir: str | Path | None = None,
    ):
        self.quantization = quantization
        self.model_path = model_path
        self.output_dir = output_dir

    def execute(self, target: Any) -> Any:
        from scomp_link.llm.core.configs import TrainResult
        from scomp_link.llm.serving.convert import ModelConverter

        path = self.model_path
        if path is None:
            if isinstance(target, TrainResult):
                path = target.model_path
            else:
                raise TypeError(
                    "LLMConvertStep requires a model path. Either pass "
                    "model_path= explicitly or chain after a step that "
                    f"returns TrainResult (got {type(target).__name__})."
                )

        mc = ModelConverter(path)
        return mc.to_gguf(
            output_dir=self.output_dir,
            quantization=self.quantization,
        )


class LLMSaveStep(LLMStep):
    """Save the upstream result as a ``.scomp`` artifact.

    Parameters
    ----------
    path : str
        Output artifact file path (e.g. ``"model.scomp"``).
    """

    def __init__(self, path: str):
        self.path = path

    def execute(self, target: Any) -> Any:
        import json

        out = Path(self.path)
        out.parent.mkdir(parents=True, exist_ok=True)

        from scomp_link.llm.core.configs import ConvertResult, TrainResult

        metadata: dict[str, Any] = {"artifact_type": "llm"}

        if isinstance(target, TrainResult):
            metadata.update(
                {
                    "kind": "train_result",
                    "model_path": str(target.model_path),
                    "adapter_path": str(target.adapter_path) if target.adapter_path else None,
                    "total_steps": target.total_steps,
                    "training_time_seconds": target.training_time_seconds,
                    "peak_memory_gb": target.peak_memory_gb,
                    "eval_loss": target.eval_loss,
                    "eval_metrics": target.eval_metrics,
                    "loss_history": target.loss_history,
                }
            )
        elif isinstance(target, ConvertResult):
            metadata.update(
                {
                    "kind": "convert_result",
                    "gguf_path": str(target.gguf_path),
                    "quantization": target.quantization,
                    "original_size_gb": target.original_size_gb,
                    "quantized_size_gb": target.quantized_size_gb,
                    "compression_ratio": target.compression_ratio,
                }
            )
        else:
            metadata["kind"] = "unknown"
            metadata["target_type"] = type(target).__name__

        out.write_text(json.dumps(metadata, indent=2))
        return str(out)


class LLMEvalStep(LLMStep):
    """Evaluate generated text quality using n-gram metrics."""

    def __init__(self, references: str | list[str] | None = None):
        self.references = references

    def execute(self, target: Any) -> Any:
        from scomp_link.llm.evaluation.quality import TextQualityMetrics

        generated = target if isinstance(target, (str, list)) else str(target)
        return TextQualityMetrics.evaluate(generated, references=self.references)


class LLMFormatStep(LLMStep):
    """Convert dataset records between instruction-tuning formats."""

    def __init__(self, source_format: str = "alpaca", target_format: str = "chatml"):
        self.source_format = source_format
        self.target_format = target_format

    def execute(self, target: Any) -> Any:
        from scomp_link.llm.data.formatting import DatasetFormatter

        if not isinstance(target, list):
            raise TypeError(f"LLMFormatStep expects a list of dicts, got {type(target).__name__}")
        return DatasetFormatter.convert_records(target, self.source_format, self.target_format)


class LLMDedupStep(LLMStep):
    """Deduplicate a list of texts."""

    def __init__(self, method: str = "exact", threshold: float = 0.8):
        self.method = method
        self.threshold = threshold

    def execute(self, target: Any) -> Any:
        from scomp_link.llm.data.dedup import TextDeduplicator

        if not isinstance(target, list):
            raise TypeError(f"LLMDedupStep expects a list of strings, got {type(target).__name__}")
        if self.method == "exact":
            clean, result = TextDeduplicator.exact_dedup(target)
        else:
            clean, result = TextDeduplicator.ngram_dedup(target, threshold=self.threshold)
        return clean


class LLMRAGBuildStep(LLMStep):
    """Build a RAG index for a directory."""

    def __init__(self, name: str, path: str, persist_dir: str = "./rag_data"):
        self.name = name
        self.path = path
        self.persist_dir = persist_dir

    def execute(self, target: Any) -> Any:
        from scomp_link.llm.rag.pipeline import RAGPipeline

        pipeline = RAGPipeline(persist_dir=self.persist_dir)
        return pipeline.build_index(self.name, self.path)


class LLMMergeStep(LLMStep):
    """Merge multiple fine-tuned models."""

    def __init__(self, base_model: str, method: str = "ties", density: float = 0.5, output_dir: str = "./merged"):
        self.base_model = base_model
        self.method = method
        self.density = density
        self.output_dir = output_dir

    def execute(self, target: Any) -> Any:
        from scomp_link.llm.serving.merge import ModelMerger

        if not isinstance(target, list):
            raise TypeError(f"LLMMergeStep expects a list of model paths, got {type(target).__name__}")
        merger = ModelMerger(self.base_model)
        method_fn = getattr(merger, f"{self.method}_merge")
        if self.method == "slerp":
            if len(target) != 2:
                raise ValueError("SLERP requires exactly 2 models")
            return method_fn(target[0], target[1], output_dir=self.output_dir)
        return method_fn(target, density=self.density, output_dir=self.output_dir)


__all__ = [
    "LLMStep",
    "LLMFineTuneStep",
    "LLMConvertStep",
    "LLMSaveStep",
    "LLMEvalStep",
    "LLMFormatStep",
    "LLMDedupStep",
    "LLMMergeStep",
    "LLMRAGBuildStep",
]


if __name__ == "__main__":
    # Demo: dedup + eval chain (no GPU needed)
    dedup = LLMDedupStep(method="exact")
    eval_step = LLMEvalStep()

    texts = ["hello world", "foo bar", "hello world", "baz qux"]
    clean = dedup.execute(texts)
    print(f"Dedup: {len(texts)} → {len(clean)} texts")

    metrics = eval_step.execute("The quick brown fox jumps over the lazy dog")
    print(f"Eval: diversity_1={metrics['diversity_1']:.3f}, rep={metrics['repetition_rate']:.3f}")

    # Chain them with >>
    chain = LLMDedupStep() >> LLMEvalStep()
    print(f"\nChain type: {type(chain).__name__}, steps: {len(chain._steps)}")
