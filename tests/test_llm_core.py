# -*- coding: utf-8 -*-
"""Tests for scomp_link/llm/core/ — configs, factory, registry, callbacks."""

import io
from pathlib import Path
from unittest.mock import patch

import pytest

from scomp_link.exceptions import DataValidationError
from scomp_link.llm.core.callbacks import Callback, PrintCallback, WandbCallback
from scomp_link.llm.core.configs import (
    ConvertResult,
    FineTuneConfig,
    TrainResult,
    TransformerConfig,
)
from scomp_link.llm.core.factory import LLMFactory
from scomp_link.llm.core.registry import get_capability

# ── TransformerConfig ────────────────────────────────────────────────────────


class TestTransformerConfig:
    def test_valid_default(self):
        cfg = TransformerConfig()
        assert cfg.d_model == 768
        assert cfg.n_heads == 12
        assert cfg.rope is True

    def test_valid_custom(self):
        cfg = TransformerConfig(d_model=256, n_heads=4)
        assert cfg.d_model == 256
        assert cfg.n_heads == 4

    def test_invalid_d_model_not_divisible(self):
        with pytest.raises(DataValidationError, match="divisible"):
            TransformerConfig(d_model=100, n_heads=3)

    def test_invalid_vocab_size_zero(self):
        with pytest.raises(DataValidationError, match="vocab_size"):
            TransformerConfig(vocab_size=0)

    def test_invalid_vocab_size_negative(self):
        with pytest.raises(DataValidationError, match="vocab_size"):
            TransformerConfig(vocab_size=-1)

    def test_invalid_dropout_too_high(self):
        with pytest.raises(DataValidationError, match="dropout"):
            TransformerConfig(dropout=1.0)

    def test_invalid_dropout_negative(self):
        with pytest.raises(DataValidationError, match="dropout"):
            TransformerConfig(dropout=-0.1)

    def test_invalid_d_model_zero(self):
        with pytest.raises(DataValidationError, match="d_model"):
            TransformerConfig(d_model=0)

    def test_invalid_n_heads_zero(self):
        with pytest.raises(DataValidationError, match="n_heads"):
            TransformerConfig(n_heads=0)


# ── FineTuneConfig ───────────────────────────────────────────────────────────


class TestFineTuneConfig:
    def test_valid_default(self):
        cfg = FineTuneConfig()
        assert cfg.method == "lora"
        assert cfg.lora_r == 16

    def test_valid_qlora(self):
        cfg = FineTuneConfig(method="qlora", bits=4)
        assert cfg.method == "qlora"
        assert cfg.bits == 4

    def test_valid_full(self):
        cfg = FineTuneConfig(method="full")
        assert cfg.method == "full"

    def test_invalid_grad_accum_zero(self):
        with pytest.raises(DataValidationError, match="gradient_accumulation_steps"):
            FineTuneConfig(gradient_accumulation_steps=0)

    def test_invalid_lora_r_zero(self):
        with pytest.raises(DataValidationError, match="lora_r"):
            FineTuneConfig(method="lora", lora_r=0)

    def test_invalid_lora_r_negative(self):
        with pytest.raises(DataValidationError, match="lora_r"):
            FineTuneConfig(method="qlora", lora_r=-1)

    def test_invalid_bits(self):
        with pytest.raises(DataValidationError, match="bits"):
            FineTuneConfig(method="qlora", bits=3)

    def test_bits_ignored_for_full(self):
        cfg = FineTuneConfig(method="full", bits=3)
        assert cfg.bits == 3


# ── TrainResult / ConvertResult ──────────────────────────────────────────────


class TestTrainResult:
    def test_create(self):
        r = TrainResult(
            loss_history=[3.0, 2.5, 2.0],
            eval_loss=1.8,
            eval_metrics={"accuracy": 0.9},
            model_path=Path("/tmp/model"),
            adapter_path=None,
            total_steps=300,
            training_time_seconds=120.5,
            peak_memory_gb=4.2,
            config=TransformerConfig(),
        )
        assert r.total_steps == 300
        assert r.eval_loss == 1.8
        assert r.adapter_path is None
        assert len(r.loss_history) == 3


class TestConvertResult:
    def test_create(self):
        r = ConvertResult(
            gguf_path=Path("/tmp/model.gguf"),
            quantization="q4_0",
            original_size_gb=13.5,
            quantized_size_gb=3.8,
            compression_ratio=3.55,
        )
        assert r.quantization == "q4_0"
        assert r.compression_ratio == pytest.approx(3.55)


# ── Factory ──────────────────────────────────────────────────────────────────


class TestFactory:
    def test_list_capabilities(self):
        caps = LLMFactory.list_capabilities()
        assert isinstance(caps, list)
        assert len(caps) >= 9
        assert caps == sorted(caps)

    def test_create_evaluate(self):
        from scomp_link.llm.evaluation.quality import TextQualityMetrics

        obj = LLMFactory.create("evaluate")
        assert isinstance(obj, TextQualityMetrics)

    def test_create_unknown(self):
        with pytest.raises(ValueError, match="Unknown LLM capability"):
            LLMFactory.create("nonexistent")

    def test_register_custom(self):
        LLMFactory.register("_test_custom", ".core.configs", "TransformerConfig")
        obj = LLMFactory.create("_test_custom", d_model=128, n_heads=4)
        assert obj.d_model == 128

    def test_register_overwrite(self):
        LLMFactory.register("_test_ow", ".core.configs", "TransformerConfig")
        LLMFactory.register("_test_ow", ".core.configs", "FineTuneConfig")
        obj = LLMFactory.create("_test_ow")
        assert isinstance(obj, FineTuneConfig)


# ── Registry ─────────────────────────────────────────────────────────────────


class TestRegistry:
    def test_get_capability_evaluate(self):
        from scomp_link.llm.evaluation.quality import TextQualityMetrics

        klass = get_capability("evaluate")
        assert klass is TextQualityMetrics

    def test_get_capability_unknown(self):
        with pytest.raises(KeyError):
            get_capability("does_not_exist")


# ── Callbacks ────────────────────────────────────────────────────────────────


class TestCallbacks:
    def test_print_callback_on_step(self, capsys):
        cb = PrintCallback()
        cb.on_step(1, 2.5)
        assert "step 1" in capsys.readouterr().out

    def test_print_callback_on_epoch_with_eval(self, capsys):
        cb = PrintCallback()
        cb.on_epoch(0, eval_loss=1.25)
        out = capsys.readouterr().out
        assert "epoch 0" in out
        assert "1.25" in out

    def test_print_callback_on_epoch_no_eval(self, capsys):
        cb = PrintCallback()
        cb.on_epoch(1, eval_loss=None)
        out = capsys.readouterr().out
        assert "epoch 1" in out
        assert "completed" in out

    def test_callback_protocol(self):
        assert isinstance(PrintCallback(), Callback)

    def test_wandb_callback_import_error(self):
        with patch.dict("sys.modules", {"wandb": None}):
            with pytest.raises(ImportError, match="wandb"):
                WandbCallback()


# ── Factory (additional) ─────────────────────────────────────────────────────


class TestFactoryExtra:
    def test_create_format(self):
        obj = LLMFactory.create("format")
        assert hasattr(obj, "from_alpaca")

    def test_factory_bad_module(self):
        LLMFactory.register("_broken", ".nonexistent.module", "FakeClass")
        with pytest.raises(ImportError):
            LLMFactory.create("_broken")
