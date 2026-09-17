# -*- coding: utf-8 -*-
"""Tests for scomp_link.llm.data (formatting, dedup) and serving.convert."""

import json
import sys
from unittest.mock import MagicMock, patch

import pytest

from scomp_link.exceptions import DataValidationError
from scomp_link.llm.data.dedup import TextDeduplicator, TextFilter
from scomp_link.llm.data.formatting import Conversation, DatasetFormatter
from scomp_link.llm.serving.convert import (
    SUPPORTED_QUANTIZATIONS,
    ModelConverter,
    _validate_quantization,
)

# ---------------------------------------------------------------------------
# Fake HuggingFace Dataset (no real 'datasets' package needed)
# ---------------------------------------------------------------------------


class _FakeDataset:
    """Mimics a HuggingFace Dataset just enough for load_dataset tests."""

    def __init__(self, data: dict):
        self._data = data
        self._len = len(next(iter(data.values()))) if data else 0

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        if isinstance(idx, int):
            return {k: v[idx] for k, v in self._data.items()}
        return self._data.get(idx, [])

    @property
    def column_names(self):
        return list(self._data.keys())

    @classmethod
    def from_pandas(cls, df):
        return cls({col: df[col].tolist() for col in df.columns})


def _patch_datasets():
    """Context manager that injects a fake 'datasets' module into sys.modules."""
    mod = MagicMock()
    mod.Dataset = _FakeDataset
    return patch.dict("sys.modules", {"datasets": mod})


# ── TestFormatting ───────────────────────────────────────────────────────────


class TestFormatting:
    def test_from_alpaca_basic(self):
        conv = DatasetFormatter.from_alpaca({"instruction": "Say hi", "output": "Hi!"})
        assert len(conv.messages) == 2
        assert conv.messages[0]["role"] == "user"
        assert conv.messages[1]["role"] == "assistant"
        assert conv.messages[1]["content"] == "Hi!"

    def test_from_alpaca_with_input(self):
        conv = DatasetFormatter.from_alpaca({"instruction": "Translate", "input": "Hello", "output": "Bonjour"})
        assert "Hello" in conv.messages[0]["content"]
        assert "Translate" in conv.messages[0]["content"]

    def test_from_alpaca_missing_key(self):
        with pytest.raises(DataValidationError, match="instruction"):
            DatasetFormatter.from_alpaca({"output": "Hi!"})

    def test_from_sharegpt(self):
        record = {
            "conversations": [
                {"from": "human", "value": "Hi"},
                {"from": "gpt", "value": "Hello!"},
            ]
        }
        conv = DatasetFormatter.from_sharegpt(record)
        assert conv.messages[0]["role"] == "user"
        assert conv.messages[1]["role"] == "assistant"

    def test_from_sharegpt_unknown_role(self):
        record = {"conversations": [{"from": "alien", "value": "beep"}]}
        with pytest.raises(DataValidationError, match="Unknown ShareGPT role"):
            DatasetFormatter.from_sharegpt(record)

    def test_from_openai(self):
        record = {
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
            ]
        }
        conv = DatasetFormatter.from_openai(record)
        assert len(conv.messages) == 3
        assert conv.messages[0]["role"] == "system"

    def test_to_chatml(self):
        conv = Conversation(messages=[{"role": "user", "content": "Hi"}])
        out = DatasetFormatter.to_chatml(conv)
        assert "<|im_start|>" in out
        assert "<|im_end|>" in out

    def test_to_llama(self):
        conv = Conversation(messages=[{"role": "user", "content": "Hi"}])
        out = DatasetFormatter.to_llama(conv)
        assert "<|begin_of_text|>" in out
        assert "<|start_header_id|>" in out

    def test_to_alpaca(self):
        conv = Conversation(
            messages=[
                {"role": "user", "content": "Do X"},
                {"role": "assistant", "content": "Done"},
            ]
        )
        out = DatasetFormatter.to_alpaca(conv)
        assert "### Instruction:" in out
        assert "### Response:" in out

    def test_to_plain(self):
        conv = Conversation(
            messages=[
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi"},
            ]
        )
        out = DatasetFormatter.to_plain(conv)
        assert "User:" in out
        assert "Assistant:" in out

    def test_convert_records(self):
        records = [
            {"instruction": "A", "output": "B"},
            {"instruction": "C", "output": "D"},
        ]
        results = DatasetFormatter.convert_records(records, "alpaca", "chatml")
        assert len(results) == 2
        assert all("<|im_start|>" in r for r in results)

    def test_convert_file(self, tmp_path):
        records = [
            {"instruction": "Say hi", "output": "Hi!"},
            {"instruction": "Count", "input": "1,2", "output": "Two"},
        ]
        src = tmp_path / "input.json"
        src.write_text(json.dumps(records))
        dst = tmp_path / "output.jsonl"

        count = DatasetFormatter.convert_file(src, dst, "alpaca", "chatml")
        assert count == 2
        assert dst.exists()
        lines = dst.read_text().strip().splitlines()
        assert len(lines) == 2
        for line in lines:
            obj = json.loads(line)
            assert "text" in obj


# ── TestDedup ────────────────────────────────────────────────────────────────


class TestDedup:
    def test_exact_dedup(self):
        texts = ["a", "b", "a", "c", "b"]
        clean, result = TextDeduplicator.exact_dedup(texts)
        assert result.original_count == 5
        assert result.deduplicated_count == 3
        assert result.duplicates_removed == 2

    def test_exact_dedup_preserves_order(self):
        texts = ["first", "second", "first", "third"]
        clean, _ = TextDeduplicator.exact_dedup(texts)
        assert clean == ["first", "second", "third"]

    def test_exact_dedup_empty(self):
        with pytest.raises(DataValidationError, match="empty"):
            TextDeduplicator.exact_dedup([])

    def test_ngram_dedup_near_duplicates(self):
        texts = [
            "The quick brown fox jumps over the lazy dog",
            "The quick brown fox jumped over the lazy dog",
            "Something completely different about cats and dogs playing",
        ]
        clean, result = TextDeduplicator.ngram_dedup(texts, threshold=0.7)
        assert result.deduplicated_count < result.original_count

    def test_ngram_dedup_different(self):
        texts = [
            "Python is a programming language",
            "The weather is nice today in Rome",
            "Mathematics explores abstract structures",
        ]
        clean, result = TextDeduplicator.ngram_dedup(texts, threshold=0.9)
        assert result.deduplicated_count == 3

    def test_filter_by_length(self):
        short_text = "too short"
        good_text = " ".join(f"word{i}" for i in range(20))  # 20 words, ~100 chars
        long_text = "x " * 60_000  # 120k chars
        texts = [short_text, good_text, long_text]
        kept, stats = TextFilter.filter_by_length(texts, min_chars=50, min_words=10, max_chars=100_000)
        assert len(kept) == 1
        assert stats["too_short"] == 1
        assert stats["too_long"] == 1

    def test_filter_by_quality_low_diversity(self):
        texts = ["the " * 50]
        kept, stats = TextFilter.filter_by_quality(texts, min_unique_words_ratio=0.3)
        assert len(kept) == 0
        assert stats["reasons"].get("low_diversity", 0) > 0

    def test_filter_by_quality_high_repetition(self):
        phrases = [
            "cat dog fish",
            "red blue green",
            "sun moon star",
            "hot cold warm",
            "big mid small",
            "one two three",
            "pen ink pad",
            "cup mug jar",
            "hat cap lid",
            "box bag can",
        ]
        repeated = " ".join(phrases * 10)
        kept, stats = TextFilter.filter_by_quality([repeated], max_repetition_rate=0.3)
        assert len(kept) == 0
        assert stats["reasons"].get("high_repetition", 0) > 0

    def test_full_pipeline(self):
        texts = [
            "A good sentence with plenty of unique interesting words here today",
            "A good sentence with plenty of unique interesting words here today",
            "x",
        ]
        kept, summary = TextFilter.full_pipeline(texts, dedup=True, min_chars=10, min_words=5)
        assert summary["original_count"] == 3
        assert summary["dedup"]["duplicates_removed"] == 1
        assert summary["final_count"] <= 2


# ── TestModelConverter ───────────────────────────────────────────────────────


class TestModelConverter:
    def test_init_missing_dir(self):
        with pytest.raises(DataValidationError, match="does not exist"):
            ModelConverter("/nonexistent/path/to/model")

    def test_init_missing_config_json(self, tmp_path):
        d = tmp_path / "model"
        d.mkdir()
        (d / "model.safetensors").write_bytes(b"\x00" * 64)
        with pytest.raises(DataValidationError, match="config.json"):
            ModelConverter(d)

    def test_init_missing_weights(self, tmp_path):
        d = tmp_path / "model"
        d.mkdir()
        (d / "config.json").write_text('{"model_type":"test"}')
        with pytest.raises(DataValidationError, match="weight files"):
            ModelConverter(d)

    def test_init_valid(self, tmp_path):
        d = tmp_path / "model"
        d.mkdir()
        (d / "config.json").write_text('{"model_type":"test"}')
        (d / "model.safetensors").write_bytes(b"\x00" * 1024)
        mc = ModelConverter(d)
        assert mc._model_path == d.resolve()

    def test_estimate_size(self, tmp_path):
        d = tmp_path / "model"
        d.mkdir()
        (d / "config.json").write_text('{"model_type":"test"}')
        # 100MB of weight data so estimate_size rounds to a nonzero GB value
        (d / "model.safetensors").write_bytes(b"\x00" * (100 * 1024 * 1024))
        mc = ModelConverter(d)
        est = mc.estimate_size("Q4_K_M")
        assert isinstance(est, float)
        assert est > 0.0

    def test_invalid_quantization(self, tmp_path):
        d = tmp_path / "model"
        d.mkdir()
        (d / "config.json").write_text('{"model_type":"test"}')
        (d / "model.safetensors").write_bytes(b"\x00" * 64)
        mc = ModelConverter(d)
        with pytest.raises(DataValidationError, match="Unsupported quantization"):
            mc.estimate_size("FAKE_QUANT")

    def test_supported_quantizations(self):
        expected = {
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
        assert SUPPORTED_QUANTIZATIONS == expected


# ── TestLoader ───────────────────────────────────────────────────────────────


class TestLoader:
    def test_load_csv(self, tmp_path):
        import pandas as pd

        from scomp_link.llm.data.loader import load_dataset

        p = tmp_path / "data.csv"
        pd.DataFrame({"text": ["hello", "world", "foo"]}).to_csv(p, index=False)
        with _patch_datasets():
            ds = load_dataset(str(p), text_field="text")
            assert len(ds) == 3
            assert "text" in ds.column_names

    def test_load_json(self, tmp_path):
        from scomp_link.llm.data.loader import load_dataset

        p = tmp_path / "data.json"
        p.write_text(json.dumps([{"text": "a"}, {"text": "b"}]))
        with _patch_datasets():
            ds = load_dataset(str(p), text_field="text")
            assert len(ds) == 2

    def test_load_jsonl(self, tmp_path):
        from scomp_link.llm.data.loader import load_dataset

        p = tmp_path / "data.jsonl"
        p.write_text('{"text":"hello"}\n{"text":"world"}\n')
        with _patch_datasets():
            ds = load_dataset(str(p), text_field="text")
            assert len(ds) == 2

    def test_load_parquet(self, tmp_path):
        import pandas as pd

        from scomp_link.llm.data.loader import load_dataset

        p = tmp_path / "data.parquet"
        pd.DataFrame({"text": ["a", "b", "c"]}).to_parquet(p)
        with _patch_datasets():
            ds = load_dataset(str(p))
            assert len(ds) == 3

    def test_load_dataframe(self):
        import pandas as pd

        from scomp_link.llm.data.loader import load_dataset

        df = pd.DataFrame({"text": ["hello", "world"]})
        with _patch_datasets():
            ds = load_dataset(df, text_field="text")
            assert len(ds) == 2

    def test_load_unsupported_format(self, tmp_path):
        from scomp_link.llm.data.loader import load_dataset

        p = tmp_path / "data.xlsx"
        p.write_text("fake")
        with _patch_datasets():
            with pytest.raises(DataValidationError, match="Unsupported file format"):
                load_dataset(str(p))

    def test_load_empty_csv(self, tmp_path):
        from scomp_link.llm.data.loader import load_dataset

        p = tmp_path / "empty.csv"
        p.write_text("text\n")
        with _patch_datasets():
            with pytest.raises(DataValidationError, match="empty"):
                load_dataset(str(p))

    def test_load_missing_column(self, tmp_path):
        from scomp_link.llm.data.loader import load_dataset

        p = tmp_path / "data.csv"
        p.write_text("col_a,col_b\n1,2\n3,4\n")
        with _patch_datasets():
            with pytest.raises(DataValidationError, match="missing required column"):
                load_dataset(str(p), text_field="text")

    def test_load_file_not_found(self):
        from scomp_link.llm.data.loader import load_dataset

        with _patch_datasets():
            with pytest.raises(DataValidationError, match="not found"):
                load_dataset("/nonexistent/data.csv")

    def test_load_unsupported_type(self):
        from scomp_link.llm.data.loader import load_dataset

        with _patch_datasets():
            with pytest.raises(DataValidationError, match="Unsupported dataset type"):
                load_dataset(12345)  # type: ignore[arg-type]
