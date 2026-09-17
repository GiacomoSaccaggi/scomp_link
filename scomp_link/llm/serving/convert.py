# -*- coding: utf-8 -*-
"""
 ██████╗ ██████╗ ███╗   ██╗██╗   ██╗███████╗██████╗ ████████╗
██╔════╝██╔═══██╗████╗  ██║██║   ██║██╔════╝██╔══██╗╚══██╔══╝
██║     ██║   ██║██╔██╗ ██║██║   ██║█████╗  ██████╔╝   ██║
██║     ██║   ██║██║╚██╗██║╚██╗ ██╔╝██╔══╝  ██╔══██╗   ██║
╚██████╗╚██████╔╝██║ ╚████║ ╚████╔╝ ███████╗██║  ██║   ██║
 ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝  ╚═══╝  ╚══════╝╚═╝  ╚═╝   ╚═╝

Convert HuggingFace models to GGUF with optional quantization (Q4_K_M, Q8_0, etc.).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Literal

from scomp_link.exceptions import DataValidationError
from scomp_link.llm.core.configs import ConvertResult

SUPPORTED_QUANTIZATIONS: set[str] = {
    "f16",
    "Q2_K",
    "Q3_K_S",
    "Q3_K_M",
    "Q3_K_L",
    "Q4_0",
    "Q4_K_S",
    "Q4_K_M",
    "Q5_0",
    "Q5_K_S",
    "Q5_K_M",
    "Q6_K",
    "Q8_0",
    "IQ2_XXS",
    "IQ2_XS",
}

# Approximate ratio of quantized size to fp16 size
_QUANT_SIZE_RATIOS: dict[str, float] = {
    "f16": 1.0,
    "Q2_K": 0.15,
    "Q3_K_S": 0.20,
    "Q3_K_M": 0.22,
    "Q3_K_L": 0.24,
    "Q4_0": 0.28,
    "Q4_K_S": 0.30,
    "Q4_K_M": 0.33,
    "Q5_0": 0.38,
    "Q5_K_S": 0.40,
    "Q5_K_M": 0.42,
    "Q6_K": 0.50,
    "Q8_0": 0.55,
    "IQ2_XXS": 0.12,
    "IQ2_XS": 0.14,
}

_SUBPROCESS_TIMEOUT = 1800  # 30 minutes

_WEIGHT_EXTENSIONS = (".safetensors", ".bin")


def _validate_quantization(quantization: str) -> None:
    if quantization not in SUPPORTED_QUANTIZATIONS:
        valid = ", ".join(sorted(SUPPORTED_QUANTIZATIONS))
        raise DataValidationError(f"Unsupported quantization type {quantization!r}. " f"Valid types: {valid}")


def _model_weight_size(model_path: Path) -> float:
    """Return total size in bytes of all weight files in *model_path*."""
    total = 0
    for ext in _WEIGHT_EXTENSIONS:
        for f in model_path.glob(f"*{ext}"):
            total += f.stat().st_size
    return float(total)


def _find_convert_script() -> str:
    """Locate ``convert_hf_to_gguf.py`` shipped with llama-cpp-python."""
    try:
        import llama_cpp  # noqa: F401
    except ImportError:
        raise ImportError("ModelConverter requires llama-cpp-python. " "Install with: pip install scomp-link[llm]")
    assert llama_cpp.__file__ is not None, "llama_cpp.__file__ is None"
    pkg_path = Path(llama_cpp.__file__).resolve().parent
    for candidate in (
        pkg_path / "convert_hf_to_gguf.py",
        pkg_path.parent / "convert_hf_to_gguf.py",
        pkg_path.parent / "scripts" / "convert_hf_to_gguf.py",
    ):
        if candidate.is_file():
            return str(candidate)
    raise ImportError(
        "Could not locate convert_hf_to_gguf.py in the llama-cpp-python "
        "installation. Ensure llama-cpp-python is installed correctly."
    )


def _find_quantize_binary() -> str:
    """Find the ``llama-quantize`` binary on PATH."""
    binary = shutil.which("llama-quantize")
    if binary is None:
        raise ImportError("Could not find 'llama-quantize' on PATH. " "Install with: pip install scomp-link[llm]")
    return binary


def _run_subprocess(cmd: list[str], *, timeout: int = _SUBPROCESS_TIMEOUT) -> None:
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"Subprocess timed out after {timeout}s: {' '.join(cmd)}") from exc

    if result.returncode != 0:
        raise RuntimeError(
            f"Subprocess failed with exit code {result.returncode}.\n"
            f"Command: {' '.join(cmd)}\n"
            f"stderr: {result.stderr.strip()}"
        )


class ModelConverter:
    """Convert HuggingFace models to GGUF with optional quantization."""

    def __init__(self, model_path: str | Path) -> None:
        self._model_path = Path(model_path).resolve()

        if not self._model_path.is_dir():
            raise DataValidationError(f"Model path does not exist or is not a directory: {self._model_path}")

        config_json = self._model_path / "config.json"
        if not config_json.is_file():
            raise DataValidationError(f"Model directory must contain config.json. " f"Not found in {self._model_path}")

        has_weights = any(any(self._model_path.glob(f"*{ext}")) for ext in _WEIGHT_EXTENSIONS)
        if not has_weights:
            raise DataValidationError(
                f"Model directory must contain weight files "
                f"(.safetensors or .bin). None found in {self._model_path}"
            )

    def to_gguf(
        self,
        output_dir: str | Path | None = None,
        quantization: str = "Q4_K_M",
        importance_matrix: str | Path | None = None,
    ) -> ConvertResult:
        _validate_quantization(quantization)

        if importance_matrix is not None:
            importance_matrix = Path(importance_matrix)
            if not importance_matrix.is_file():
                raise DataValidationError(f"importance_matrix file not found: {importance_matrix}")

        convert_script = _find_convert_script()
        quantize_bin = _find_quantize_binary() if quantization != "f16" else None

        out = Path(output_dir) if output_dir is not None else self._model_path / "gguf"
        out.mkdir(parents=True, exist_ok=True)

        model_name = self._model_path.name
        fp16_path = out / f"{model_name}-f16.gguf"

        # Step 1: HF → fp16 GGUF
        _run_subprocess(
            [
                sys.executable,
                convert_script,
                str(self._model_path),
                "--outtype",
                "f16",
                "--outfile",
                str(fp16_path),
            ]
        )
        if not fp16_path.is_file():
            raise RuntimeError(f"Conversion produced no output file at {fp16_path}")

        # Step 2: Quantize (skip if target is f16)
        if quantization == "f16":
            final_path = fp16_path
        else:
            final_path = out / f"{model_name}-{quantization}.gguf"
            cmd: list[str] = [quantize_bin]  # type: ignore[list-item]
            if importance_matrix is not None:
                cmd += ["--imatrix", str(importance_matrix)]
            cmd += [str(fp16_path), str(final_path), quantization]

            try:
                _run_subprocess(cmd)
            except Exception:
                if final_path.is_file():
                    final_path.unlink()
                raise

            # Clean up intermediate fp16
            if fp16_path.is_file():
                fp16_path.unlink()

        original_size = _model_weight_size(self._model_path)
        quantized_size = float(final_path.stat().st_size)

        return ConvertResult(
            gguf_path=final_path,
            quantization=quantization,
            original_size_gb=round(original_size / (1024**3), 2),
            quantized_size_gb=round(quantized_size / (1024**3), 2),
            compression_ratio=round(
                original_size / quantized_size if quantized_size > 0 else 0.0,
                2,
            ),
        )

    def estimate_size(self, quantization: str) -> float:
        _validate_quantization(quantization)
        weight_bytes = _model_weight_size(self._model_path)
        ratio = _QUANT_SIZE_RATIOS[quantization]
        return round(weight_bytes * ratio / (1024**3), 2)

    def push_to_hub(self, gguf_path: str | Path, repo_id: str) -> str:
        try:
            from huggingface_hub import HfApi
        except ImportError:
            raise ImportError("push_to_hub requires huggingface_hub. " "Install with: pip install scomp-link[llm]")

        gguf_path = Path(gguf_path)
        if not gguf_path.is_file():
            raise DataValidationError(f"GGUF file not found: {gguf_path}")

        api = HfApi()
        api.create_repo(repo_id, exist_ok=True, repo_type="model")
        url = api.upload_file(
            path_or_fileobj=str(gguf_path),
            path_in_repo=gguf_path.name,
            repo_id=repo_id,
        )
        return str(url)


if __name__ == "__main__":
    import json
    import tempfile
    from pathlib import Path

    # Create a fake model directory to test validation
    with tempfile.TemporaryDirectory() as d:
        model_dir = Path(d) / "fake_model"
        model_dir.mkdir()
        (model_dir / "config.json").write_text(json.dumps({"model_type": "test"}))
        (model_dir / "model.safetensors").write_bytes(b"\x00" * 1024)

        mc = ModelConverter(model_dir)
        est = mc.estimate_size("Q4_K_M")
        print(f"Model dir: {model_dir.name}")
        print(f"  Estimated Q4_K_M size: {est:.4f} GB")
        print(f"  Supported quants: {', '.join(sorted(SUPPORTED_QUANTIZATIONS))}")

    # Test validation
    try:
        ModelConverter("/nonexistent/path")
    except DataValidationError as e:
        print(f"  Validation OK: {e}")
