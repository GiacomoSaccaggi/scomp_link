# -*- coding: utf-8 -*-
"""
███╗   ███╗███████╗██████╗  ██████╗ ███████╗
████╗ ████║██╔════╝██╔══██╗██╔════╝ ██╔════╝
██╔████╔██║█████╗  ██████╔╝██║  ███╗█████╗
██║╚██╔╝██║██╔══╝  ██╔══██╗██║   ██║██╔══╝
██║ ╚═╝ ██║███████╗██║  ██║╚██████╔╝███████╗
╚═╝     ╚═╝╚══════╝╚═╝  ╚═╝ ╚═════╝ ╚══════╝

Model merging: linear, SLERP, TIES, and DARE strategies for combining fine-tunes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

from scomp_link.exceptions import DataValidationError

logger = logging.getLogger(__name__)

_INSTALL_MSG = "ModelMerger requires torch and transformers. " "Install with: pip install scomp-link[llm]"


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


def _load_state_dict(model_path: str | Path) -> dict:
    torch = _require_torch()
    _require_transformers()
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(str(model_path), torch_dtype=torch.float32, device_map="cpu")
    sd = model.state_dict()
    del model
    return sd


class ModelMerger:
    """Merge multiple HuggingFace models using various strategies."""

    def __init__(self, base_model: str | Path) -> None:
        _require_torch()
        _require_transformers()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._base_path = str(base_model)
        self._tokenizer = AutoTokenizer.from_pretrained(self._base_path, trust_remote_code=True)
        self._base_sd = _load_state_dict(self._base_path)

    def linear_merge(
        self,
        models: list[str | Path],
        weights: list[float] | None = None,
        output_dir: str | Path = "./merged",
    ) -> Path:
        """Weighted average of model parameters."""
        _require_torch()

        if len(models) < 2:
            raise DataValidationError("linear_merge requires at least 2 models")
        if weights is None:
            weights = [1.0 / len(models)] * len(models)
        if len(weights) != len(models):
            raise DataValidationError(f"weights length ({len(weights)}) must match models length ({len(models)})")
        w_sum = sum(weights)
        if abs(w_sum - 1.0) > 1e-6:
            weights = [w / w_sum for w in weights]

        state_dicts = [_load_state_dict(m) for m in models]
        merged_sd = {}
        for key in self._base_sd:
            merged_sd[key] = sum(w * sd[key].float() for w, sd in zip(weights, state_dicts))

        return self._save_merged(merged_sd, output_dir)

    def slerp_merge(
        self,
        model_a: str | Path,
        model_b: str | Path,
        t: float = 0.5,
        output_dir: str | Path = "./merged",
    ) -> Path:
        """Spherical Linear Interpolation between two models."""
        torch = _require_torch()

        if not 0.0 <= t <= 1.0:
            raise DataValidationError(f"t must be in [0.0, 1.0], got {t}")

        sd_a = _load_state_dict(model_a)
        sd_b = _load_state_dict(model_b)
        merged_sd = {}

        for key in self._base_sd:
            a = sd_a[key].float().flatten()
            b = sd_b[key].float().flatten()

            norm_a = torch.linalg.norm(a)
            norm_b = torch.linalg.norm(b)

            if norm_a < 1e-8 or norm_b < 1e-8:
                merged_sd[key] = ((1.0 - t) * sd_a[key].float() + t * sd_b[key].float()).reshape(sd_a[key].shape)
                continue

            cos_omega = torch.clamp(torch.dot(a, b) / (norm_a * norm_b), -1.0, 1.0)

            if cos_omega.abs() > 1.0 - 1e-6:
                merged_sd[key] = (1.0 - t) * sd_a[key].float() + t * sd_b[key].float()
                continue

            omega = torch.acos(cos_omega)
            sin_omega = torch.sin(omega)
            coeff_a = torch.sin((1.0 - t) * omega) / sin_omega
            coeff_b = torch.sin(t * omega) / sin_omega
            merged_sd[key] = coeff_a * sd_a[key].float() + coeff_b * sd_b[key].float()

        return self._save_merged(merged_sd, output_dir)

    def ties_merge(
        self,
        models: list[str | Path],
        density: float = 0.5,
        output_dir: str | Path = "./merged",
    ) -> Path:
        """TIES-Merging: Trim, Elect Sign, Disjoint Merge."""
        torch = _require_torch()

        if len(models) < 2:
            raise DataValidationError("ties_merge requires at least 2 models")
        if not 0.0 < density <= 1.0:
            raise DataValidationError(f"density must be in (0.0, 1.0], got {density}")

        state_dicts = [_load_state_dict(m) for m in models]
        merged_sd = {}

        for key in self._base_sd:
            base_param = self._base_sd[key].float()
            task_vectors = [sd[key].float() - base_param for sd in state_dicts]

            # 1. Trim: zero out smallest (1-density) fraction per task vector
            trimmed = []
            for tv in task_vectors:
                flat = tv.flatten()
                k = max(1, int(density * flat.numel()))
                threshold = flat.abs().topk(k).values[-1]
                mask = flat.abs() >= threshold
                trimmed.append((tv * mask.reshape(tv.shape)))

            # 2. Elect sign: majority vote across models
            stacked = torch.stack(trimmed)
            sign_sum = torch.sign(stacked).sum(dim=0)
            elected_sign = torch.sign(sign_sum)
            elected_sign[elected_sign == 0] = 1.0

            # 3. Disjoint merge: keep only values matching elected sign, then average
            acc = torch.zeros_like(base_param)
            count = torch.zeros_like(base_param)
            for tv in trimmed:
                agree = torch.sign(tv) == elected_sign
                contrib = tv * agree
                acc += contrib
                count += (contrib != 0).float()

            count = count.clamp(min=1.0)
            merged_sd[key] = base_param + acc / count

        return self._save_merged(merged_sd, output_dir)

    def dare_merge(
        self,
        models: list[str | Path],
        density: float = 0.5,
        output_dir: str | Path = "./merged",
    ) -> Path:
        """DARE: Drop And REscale."""
        torch = _require_torch()

        if len(models) < 2:
            raise DataValidationError("dare_merge requires at least 2 models")
        if not 0.0 < density <= 1.0:
            raise DataValidationError(f"density must be in (0.0, 1.0], got {density}")

        state_dicts = [_load_state_dict(m) for m in models]
        merged_sd = {}

        for key in self._base_sd:
            base_param = self._base_sd[key].float()
            task_vectors = [sd[key].float() - base_param for sd in state_dicts]

            rescaled = []
            for tv in task_vectors:
                mask = torch.bernoulli(torch.full_like(tv, density))
                rescaled.append(tv * mask / density)

            avg_tv = torch.stack(rescaled).mean(dim=0)
            merged_sd[key] = base_param + avg_tv

        return self._save_merged(merged_sd, output_dir)

    def _save_merged(self, merged_sd: dict, output_dir: str | Path) -> Path:
        _require_transformers()
        from transformers import AutoConfig, AutoModelForCausalLM

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        config = AutoConfig.from_pretrained(self._base_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(config)
        model.load_state_dict(merged_sd, strict=True)
        model.save_pretrained(str(output_dir))
        self._tokenizer.save_pretrained(str(output_dir))
        return output_dir

    @staticmethod
    def list_methods() -> list[str]:
        return ["linear", "slerp", "ties", "dare"]


if __name__ == "__main__":
    print(f"Available methods: {', '.join(ModelMerger.list_methods())}")
    print("\nUsage:")
    print("  merger = ModelMerger('base_model_path')")
    print("  merger.ties_merge(['model_a', 'model_b'], density=0.5, output_dir='./merged')")
    print("  merger.slerp_merge('model_a', 'model_b', t=0.6)")
    print("  merger.dare_merge(['model_a', 'model_b', 'model_c'], density=0.3)")
