# -*- coding: utf-8 -*-
"""
 ██████╗ ██████╗ ███╗   ██╗███████╗██╗ ██████╗ ███████╗
██╔════╝██╔═══██╗████╗  ██║██╔════╝██║██╔════╝ ██╔════╝
██║     ██║   ██║██╔██╗ ██║█████╗  ██║██║  ███╗███████╗
██║     ██║   ██║██║╚██╗██║██╔══╝  ██║██║   ██║╚════██║
╚██████╗╚██████╔╝██║ ╚████║██║     ██║╚██████╔╝███████║
 ╚═════╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝     ╚═╝ ╚═════╝ ╚══════╝

Dataclasses for transformer config, fine-tuning config, and result containers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Union

from scomp_link.exceptions import DataValidationError


@dataclass
class TransformerConfig:
    vocab_size: int = 32_000
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    d_ff: int = 3072
    max_seq_len: int = 2048
    dropout: float = 0.1
    rope: bool = True
    flash_attention: bool = False

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise DataValidationError(f"vocab_size must be > 0, got {self.vocab_size}")
        if self.d_model < 1:
            raise DataValidationError(f"d_model must be >= 1, got {self.d_model}")
        if self.n_heads < 1:
            raise DataValidationError(f"n_heads must be >= 1, got {self.n_heads}")
        if self.d_model % self.n_heads != 0:
            raise DataValidationError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        if not (0.0 <= self.dropout < 1.0):
            raise DataValidationError(f"dropout must be in [0.0, 1.0), got {self.dropout}")


@dataclass
class FineTuneConfig:
    method: Literal["lora", "qlora", "full"] = "lora"
    # LoRA params
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] | None = None
    # Quantization (QLoRA)
    bits: Literal[4, 8] = 4
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    # Training
    mixed_precision: Literal["no", "fp16", "bf16"] = "bf16"
    gradient_accumulation_steps: int = 4
    gradient_checkpointing: bool = True
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    # Data
    max_seq_length: int = 2048
    packing: bool = False
    dataset_text_field: str = "text"
    # Output
    output_dir: str = "./llm_output"
    save_steps: int = 100
    logging_steps: int = 10

    def __post_init__(self) -> None:
        if self.gradient_accumulation_steps < 1:
            raise DataValidationError(
                f"gradient_accumulation_steps must be >= 1, got {self.gradient_accumulation_steps}"
            )
        if self.method in ("lora", "qlora") and (not isinstance(self.lora_r, int) or self.lora_r < 1):
            raise DataValidationError(
                f"lora_r must be a positive integer when method is '{self.method}', got {self.lora_r}"
            )
        if self.method == "qlora" and self.bits not in (4, 8):
            raise DataValidationError(f"bits must be 4 or 8 when method is 'qlora', got {self.bits}")


@dataclass
class TrainResult:
    loss_history: list[float]
    eval_loss: float | None
    eval_metrics: dict[str, float]
    model_path: Path
    adapter_path: Path | None
    total_steps: int
    training_time_seconds: float
    peak_memory_gb: float
    config: Union[TransformerConfig, FineTuneConfig]


@dataclass
class ConvertResult:
    gguf_path: Path
    quantization: str
    original_size_gb: float
    quantized_size_gb: float
    compression_ratio: float


if __name__ == "__main__":
    # Build configs, check defaults, trip validation on purpose
    t = TransformerConfig(d_model=256, n_heads=4, n_layers=3)
    print(f"TransformerConfig: {t.d_model}d, {t.n_heads}h, {t.n_layers}L, rope={t.rope}")

    f = FineTuneConfig(method="lora", lora_r=8, gradient_accumulation_steps=2)
    print(f"FineTuneConfig: {f.method}, rank={f.lora_r}, accum={f.gradient_accumulation_steps}")

    # These should all fail with DataValidationError
    bad_cases = [
        ("d_model not divisible by n_heads", dict(d_model=100, n_heads=3)),
        ("vocab_size zero", dict(vocab_size=0)),
        ("dropout out of range", dict(dropout=1.5)),
    ]
    for desc, kwargs in bad_cases:
        try:
            TransformerConfig(**kwargs)
            print(f"  FAIL: {desc} should have raised")
        except DataValidationError as e:
            print(f"  OK: {desc} → {e}")
