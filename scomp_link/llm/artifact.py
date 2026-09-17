# -*- coding: utf-8 -*-
"""
LLM artifact persistence — save and load TrainResult as .scomp files.

Extends the core ScompArtifact format with LLM-specific metadata:
  - Full TrainResult fields (config, loss_history, eval_metrics, …)
  - SHA-256 hashes of model weight files for integrity verification
  - adapter_path tracking for LoRA/QLoRA methods
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Union

from scomp_link.exceptions import ArtifactError
from scomp_link.persistence.artifact import ScompArtifact
from scomp_link.utils.logger import get_logger

logger = get_logger(__name__)

_WEIGHT_EXTENSIONS = frozenset(
    {
        ".safetensors",
        ".bin",
        ".pt",
        ".pth",
        ".ckpt",
    }
)


def _hash_directory(path: Path) -> dict[str, str]:
    """Compute SHA-256 hashes for all weight files in *path*."""
    hashes: dict[str, str] = {}
    if not path.is_dir():
        return hashes
    for f in sorted(path.rglob("*")):
        if f.is_file() and f.suffix in _WEIGHT_EXTENSIONS:
            h = hashlib.sha256()
            with open(f, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            hashes[str(f.relative_to(path))] = h.hexdigest()
    return hashes


def _serialize_config(cfg) -> dict:
    """Convert a TransformerConfig or FineTuneConfig dataclass to a plain dict."""
    if dataclasses.is_dataclass(cfg) and not isinstance(cfg, type):
        d = dataclasses.asdict(cfg)
        d["__dataclass__"] = type(cfg).__name__
        return d
    return {}


def _deserialize_config(d: dict):
    """Restore a TransformerConfig or FineTuneConfig from a plain dict."""
    from scomp_link.llm.core.configs import FineTuneConfig, TransformerConfig

    cls_name = d.pop("__dataclass__", None)
    mapping = {
        "TransformerConfig": TransformerConfig,
        "FineTuneConfig": FineTuneConfig,
    }
    cls = mapping.get(cls_name)
    if cls is None:
        return d
    # Only pass fields that the dataclass actually defines.
    valid = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in valid})


def save_llm_artifact(
    result,  # TrainResult — not type-annotated to keep imports lazy
    path: Union[str, Path],
    *,
    hf_model_id: str | None = None,
    hf_commit_hash: str | None = None,
) -> Path:
    """Persist a *TrainResult* as a ``.scomp`` artifact.

    Saves the full training result (config, loss_history, eval_metrics,
    total_steps, training_time_seconds, peak_memory_gb), the HuggingFace
    model identifier / commit hash, and SHA-256 hashes of all weight files
    referenced by ``model_path`` (and ``adapter_path`` for LoRA/QLoRA).

    Parameters
    ----------
    result : TrainResult
        Training result to persist.
    path : str | Path
        Destination file path (will have ``.scomp`` appended if no suffix).
    hf_model_id : str, optional
        HuggingFace model identifier used for fine-tuning.
    hf_commit_hash : str, optional
        Git commit hash of the HuggingFace model at download time.

    Returns
    -------
    Path
        The written artifact path.
    """
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(".scomp")

    artifact = ScompArtifact()

    # ── config.json: training config + provenance ────────────────────────
    config_data: dict = {
        "llm": True,
        "training_config": _serialize_config(result.config),
        "total_steps": result.total_steps,
        "training_time_seconds": result.training_time_seconds,
        "peak_memory_gb": result.peak_memory_gb,
        "model_path": str(result.model_path),
    }
    if result.adapter_path is not None:
        config_data["adapter_path"] = str(result.adapter_path)
    if hf_model_id is not None:
        config_data["hf_model_id"] = hf_model_id
    if hf_commit_hash is not None:
        config_data["hf_commit_hash"] = hf_commit_hash

    artifact.set_config(**config_data)

    # ── metrics.json: loss history + eval metrics ────────────────────────
    metrics: dict = {
        "loss_history": result.loss_history,
        "eval_metrics": result.eval_metrics,
    }
    if result.eval_loss is not None:
        metrics["eval_loss"] = result.eval_loss
    artifact.set_metrics(metrics)

    # ── metadata / manifest: SHA-256 hashes ──────────────────────────────
    weight_hashes: dict = {}
    model_path = Path(result.model_path)
    if model_path.is_dir():
        weight_hashes["model"] = _hash_directory(model_path)

    if result.adapter_path is not None:
        adapter_path = Path(result.adapter_path)
        if adapter_path.is_dir():
            weight_hashes["adapter"] = _hash_directory(adapter_path)

    artifact.set_metadata(
        llm_artifact=True,
        weight_hashes=weight_hashes,
    )

    saved = artifact.save(path)
    logger.info(f"✅ Saved LLM artifact: {saved}")
    return saved


def load_llm_artifact(path: Union[str, Path]):
    """Load a ``.scomp`` artifact and reconstruct a *TrainResult*.

    Verifies that referenced weight files still exist and that their
    SHA-256 hashes match those recorded at save time.

    Parameters
    ----------
    path : str | Path
        Path to an existing ``.scomp`` file.

    Returns
    -------
    TrainResult
        Restored training result with all persisted fields.

    Raises
    ------
    ArtifactError
        If the artifact is invalid, weight paths are missing, or hashes
        do not match.
    """
    from scomp_link.llm.core.configs import TrainResult

    path = Path(path)
    artifact = ScompArtifact.load(path)
    cfg = artifact.config
    met = artifact.metrics

    if not cfg.get("llm"):
        raise ArtifactError(f"'{path}' is not an LLM artifact (missing 'llm' flag in config).")

    # ── Restore model_path / adapter_path ────────────────────────────────
    model_path = Path(cfg["model_path"])
    adapter_path = Path(cfg["adapter_path"]) if "adapter_path" in cfg else None

    # ── Verify paths exist ───────────────────────────────────────────────
    if not model_path.exists():
        raise ArtifactError(f"model_path does not exist: {model_path}")
    if adapter_path is not None and not adapter_path.exists():
        raise ArtifactError(f"adapter_path does not exist: {adapter_path}")

    # ── Verify weight file hashes ────────────────────────────────────────
    weight_hashes = artifact.metadata.get("weight_hashes", {})

    saved_model_hashes = weight_hashes.get("model", {})
    if saved_model_hashes and model_path.is_dir():
        current = _hash_directory(model_path)
        _verify_hashes(saved_model_hashes, current, "model_path", model_path)

    if adapter_path is not None:
        saved_adapter_hashes = weight_hashes.get("adapter", {})
        if saved_adapter_hashes and adapter_path.is_dir():
            current = _hash_directory(adapter_path)
            _verify_hashes(saved_adapter_hashes, current, "adapter_path", adapter_path)

    # ── Reconstruct TrainResult ──────────────────────────────────────────
    training_config = _deserialize_config(dict(cfg.get("training_config", {})))

    return TrainResult(
        loss_history=met.get("loss_history", []),
        eval_loss=met.get("eval_loss"),
        eval_metrics=met.get("eval_metrics", {}),
        model_path=model_path,
        adapter_path=adapter_path,
        total_steps=cfg.get("total_steps", 0),
        training_time_seconds=cfg.get("training_time_seconds", 0.0),
        peak_memory_gb=cfg.get("peak_memory_gb", 0.0),
        config=training_config,  # type: ignore[arg-type]
    )


def _verify_hashes(
    saved: dict[str, str],
    current: dict[str, str],
    label: str,
    base_path: Path,
) -> None:
    """Raise ArtifactError if any saved hash doesn't match the current hash."""
    for rel_path, expected_hash in saved.items():
        actual_hash = current.get(rel_path)
        if actual_hash is None:
            raise ArtifactError(f"Weight file missing from {label}: {base_path / rel_path}")
        if actual_hash != expected_hash:
            raise ArtifactError(
                f"Weight file modified in {label}: {base_path / rel_path} "
                f"(expected SHA-256 {expected_hash[:16]}…, got {actual_hash[:16]}…)"
            )


if __name__ == "__main__":
    # Show what gets saved in a .scomp artifact
    from pathlib import Path

    from scomp_link.llm.core.configs import FineTuneConfig, TrainResult

    result = TrainResult(
        loss_history=[2.5, 1.8, 1.2, 0.9],
        eval_loss=0.95,
        eval_metrics={"perplexity": 2.59},
        model_path=Path("."),
        adapter_path=None,
        total_steps=400,
        training_time_seconds=120.5,
        peak_memory_gb=6.2,
        config=FineTuneConfig(method="lora", lora_r=16),
    )
    print(f"TrainResult: {result.total_steps} steps, loss {result.loss_history[-1]:.2f}")
    if isinstance(result.config, FineTuneConfig):
        print(f"  Config: {result.config.method}, rank={result.config.lora_r}")
    print(f"  Memory: {result.peak_memory_gb:.1f} GB, time: {result.training_time_seconds:.1f}s")
    print(f"  Eval: loss={result.eval_loss}, metrics={result.eval_metrics}")
