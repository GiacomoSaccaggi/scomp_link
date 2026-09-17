# -*- coding: utf-8 -*-
"""Tests for scomp_link.llm.dsl and scomp_link.llm.artifact."""

import json
from pathlib import Path

import pytest

from scomp_link.llm.artifact import (
    _deserialize_config,
    _hash_directory,
    _serialize_config,
)
from scomp_link.llm.core.configs import FineTuneConfig
from scomp_link.llm.dsl import (
    LLMDedupStep,
    LLMEvalStep,
    LLMFormatStep,
    LLMSaveStep,
)

# ── TestDSLSteps ─────────────────────────────────────────────────────────────


class TestDSLSteps:
    def test_eval_step_string(self):
        step = LLMEvalStep()
        result = step.execute("The quick brown fox jumps over the lazy dog")
        assert isinstance(result, dict)
        assert "diversity_1" in result
        assert "repetition_rate" in result

    def test_eval_step_list(self):
        step = LLMEvalStep()
        result = step.execute(["Hello world", "Goodbye world"])
        assert isinstance(result, dict)
        assert "diversity_1" in result

    def test_dedup_step_exact(self):
        step = LLMDedupStep(method="exact")
        result = step.execute(["a", "b", "a", "c"])
        assert result == ["a", "b", "c"]

    def test_dedup_step_ngram(self):
        step = LLMDedupStep(method="ngram", threshold=0.7)
        texts = [
            "The quick brown fox jumps over the lazy dog",
            "The quick brown fox jumped over the lazy dog",
            "Something completely different about cats",
        ]
        result = step.execute(texts)
        assert len(result) < len(texts)

    def test_format_step(self):
        step = LLMFormatStep(source_format="alpaca", target_format="chatml")
        records = [
            {"instruction": "Say hi", "output": "Hi!"},
            {"instruction": "Count", "output": "One"},
        ]
        result = step.execute(records)
        assert len(result) == 2
        assert all("<|im_start|>" in r for r in result)

    def test_format_step_wrong_input(self):
        step = LLMFormatStep()
        with pytest.raises(TypeError, match="list of dicts"):
            step.execute("not a list")

    def test_save_step(self, tmp_path):
        step = LLMSaveStep(str(tmp_path / "out.scomp"))
        result = step.execute({"some": "data"})
        p = Path(result)
        assert p.exists()
        content = json.loads(p.read_text())
        assert content["artifact_type"] == "llm"
        assert content["kind"] == "unknown"

    def test_chain_dedup_eval(self):
        chain = LLMDedupStep() >> LLMEvalStep()
        assert len(chain._steps) == 2

    def test_chain_type_safety(self):
        from scomp_link.pipeline_dsl import CleanStep

        with pytest.raises(TypeError, match="Cannot mix"):
            LLMDedupStep() >> CleanStep(None)


# ── TestArtifact ─────────────────────────────────────────────────────────────


class TestArtifact:
    def test_serialize_config(self):
        cfg = FineTuneConfig(method="lora", lora_r=8)
        d = _serialize_config(cfg)
        assert isinstance(d, dict)
        assert d["__dataclass__"] == "FineTuneConfig"
        assert d["method"] == "lora"
        assert d["lora_r"] == 8

    def test_deserialize_config(self):
        cfg = FineTuneConfig(method="qlora", lora_r=32, bits=4)
        serialized = _serialize_config(cfg)
        restored = _deserialize_config(dict(serialized))
        assert isinstance(restored, FineTuneConfig)
        assert restored.method == "qlora"
        assert restored.lora_r == 32
        assert restored.bits == 4

    def test_hash_directory(self, tmp_path):
        d = tmp_path / "weights"
        d.mkdir()
        (d / "layer1.safetensors").write_bytes(b"\xde\xad" * 512)
        (d / "layer2.bin").write_bytes(b"\xbe\xef" * 256)
        (d / "readme.txt").write_text("ignore me")
        hashes = _hash_directory(d)
        assert len(hashes) == 2
        assert "layer1.safetensors" in hashes
        assert "layer2.bin" in hashes
        assert all(len(v) == 64 for v in hashes.values())

    def test_hash_directory_empty(self, tmp_path):
        d = tmp_path / "empty"
        d.mkdir()
        hashes = _hash_directory(d)
        assert hashes == {}


# ── TestDSLSteps (additional) ────────────────────────────────────────────────


class TestDSLStepsExtra:
    def test_eval_step_with_references(self):
        step = LLMEvalStep(references="the cat is on the mat")
        result = step.execute("the cat sat on the mat")
        assert "bleu" in result
        assert "rouge_1_f1" in result

    def test_dedup_step_wrong_input(self):
        step = LLMDedupStep()
        with pytest.raises(TypeError, match="list of strings"):
            step.execute("not a list")
