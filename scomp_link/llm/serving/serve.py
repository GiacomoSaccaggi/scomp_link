# -*- coding: utf-8 -*-
"""
███████╗███████╗██████╗ ██╗   ██╗███████╗
██╔════╝██╔════╝██╔══██╗██║   ██║██╔════╝
███████╗█████╗  ██████╔╝██║   ██║█████╗
╚════██║██╔══╝  ██╔══██╗╚██╗ ██╔╝██╔══╝
███████║███████╗██║  ██║ ╚████╔╝ ███████╗
╚══════╝╚══════╝╚═╝  ╚═╝  ╚═══╝  ╚══════╝

Local inference server: load a model and expose /generate, /health, /info endpoints.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

from scomp_link.exceptions import DataValidationError, ModelTrainingError

logger = logging.getLogger(__name__)

_INSTALL_MSG = "InferenceServer requires torch and transformers. " "Install with: pip install scomp-link[llm]"


def _require_torch():
    try:
        import torch

        return torch
    except ImportError:
        raise ImportError(_INSTALL_MSG)


def _require_transformers():
    try:
        import transformers

        return transformers
    except ImportError:
        raise ImportError(_INSTALL_MSG)


class InferenceServer:
    """Serve a fine-tuned HuggingFace model locally via REST API."""

    def __init__(
        self,
        model_path: str | Path,
        device: str = "auto",
        max_length: int = 2048,
        load_in_4bit: bool = False,
    ) -> None:
        torch = _require_torch()
        _require_transformers()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_path = str(model_path)
        self.max_length = max_length
        self.load_in_4bit = load_in_4bit

        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        model_kwargs: dict = {"trust_remote_code": True}

        if load_in_4bit:
            try:
                import bitsandbytes  # noqa: F401
            except ImportError:
                raise ImportError("4-bit loading requires bitsandbytes. " "Install with: pip install bitsandbytes")
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["device_map"] = self.device if self.device != "cpu" else None

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **model_kwargs)
        except Exception as exc:
            raise ModelTrainingError(f"Failed to load model from {self.model_path!r}: {exc}") from exc

        if not load_in_4bit and self.device != "cpu":
            self.model.to(self.device)  # type: ignore[arg-type]

        self.model.eval()
        logger.info(
            "Model loaded from %s on %s (%s parameters)",
            self.model_path,
            self.device,
            f"{self._count_parameters():,}",
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        stop_sequences: list[str] | None = None,
    ) -> dict:
        torch = _require_torch()

        if not prompt:
            raise DataValidationError("prompt must be a non-empty string")
        if max_new_tokens < 1:
            raise DataValidationError("max_new_tokens must be >= 1")
        if not (0.0 < temperature <= 2.0):
            raise DataValidationError("temperature must be in (0.0, 2.0]")

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]

        gen_kwargs: dict = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": repetition_penalty,
            "do_sample": temperature > 0.0,
        }

        t0 = time.perf_counter()
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, **gen_kwargs)  # type: ignore[arg-type]
        elapsed = time.perf_counter() - t0

        new_tokens = output_ids[0][input_len:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

        if stop_sequences:
            earliest = len(text)
            for seq in stop_sequences:
                idx = text.find(seq)
                if idx != -1 and idx < earliest:
                    earliest = idx
            text = text[:earliest]

        return {
            "text": text,
            "tokens_generated": len(new_tokens),
            "time_seconds": round(elapsed, 3),
        }

    def start(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        """Start a Flask server exposing /generate, /health, and /info."""
        from flask import Flask, jsonify, request

        app = Flask(__name__)

        @app.route("/generate", methods=["POST"])
        def _generate():
            body = request.get_json(silent=True)
            if not body or "prompt" not in body:
                return jsonify({"error": "request body must include 'prompt'"}), 400
            kwargs = {
                k: body[k]
                for k in (
                    "prompt",
                    "max_new_tokens",
                    "temperature",
                    "top_k",
                    "top_p",
                    "repetition_penalty",
                    "stop_sequences",
                )
                if k in body
            }
            try:
                result = self.generate(**kwargs)
            except (DataValidationError, ModelTrainingError) as exc:
                return jsonify({"error": str(exc)}), 400
            except Exception as exc:
                logger.exception("Generation failed")
                return jsonify({"error": str(exc)}), 500
            return jsonify(result)

        @app.route("/health", methods=["GET"])
        def _health():
            return jsonify(
                {
                    "status": "ok",
                    "model": self.model_path,
                    "device": self.device,
                }
            )

        @app.route("/info", methods=["GET"])
        def _info():
            return jsonify(
                {
                    "model_path": self.model_path,
                    "device": self.device,
                    "max_length": self.max_length,
                    "parameters": self._count_parameters(),
                }
            )

        logger.info("Starting inference server on %s:%d", host, port)
        app.run(host=host, port=port)

    def _count_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())


if __name__ == "__main__":
    print("InferenceServer — local model serving via Flask REST API")
    print("\nEndpoints:")
    print("  POST /generate  → {prompt, max_new_tokens, temperature, ...}")
    print("  GET  /health    → {status, model, device}")
    print("  GET  /info      → {model_path, device, parameters}")
    print("\nUsage:")
    print("  server = InferenceServer('./my_model', load_in_4bit=True)")
    print("  server.start(port=8080)")
    print("\n  # Or just generate without starting the server:")
    print("  result = server.generate('Hello!', max_new_tokens=100)")
    print("  print(result['text'])")
