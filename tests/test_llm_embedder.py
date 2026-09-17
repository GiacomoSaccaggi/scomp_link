# -*- coding: utf-8 -*-
"""Tests for scomp_link.llm.rag.embedder — profiles only, no model download."""

from scomp_link.llm.rag.embedder import (
    _MODEL_PROFILES,
    DEFAULT_EMBED_MODEL,
    FALLBACK_EMBED_MODEL,
    LocalEmbedder,
    _get_profile,
)


class TestEmbedderProfiles:
    def test_known_model(self):
        p = _get_profile("Qwen/Qwen3-Embedding-0.6B")
        assert p["dim"] == 1024
        assert p["matryoshka"] is True
        assert p["license"] == "Apache-2.0"

    def test_partial_match(self):
        p = _get_profile("Qwen3-Embedding-0.6B")
        assert p["dim"] == 1024

    def test_unknown_model(self):
        p = _get_profile("totally/unknown-model")
        assert p["dim"] is None
        assert p["prefix_query"] == ""

    def test_all_profiles_valid(self):
        for name, profile in _MODEL_PROFILES.items():
            assert "dim" in profile
            assert "max_tokens" in profile
            assert "license" in profile

    def test_list_models(self):
        models = LocalEmbedder.list_models()
        assert len(models) == len(_MODEL_PROFILES)
        for m in models:
            assert "model" in m
            assert "dim" in m

    def test_default_and_fallback(self):
        assert DEFAULT_EMBED_MODEL in _MODEL_PROFILES
        assert FALLBACK_EMBED_MODEL in _MODEL_PROFILES

    def test_e5_profile(self):
        p = _get_profile("intfloat/multilingual-e5-small")
        assert p["dim"] == 384
        assert p["prefix_query"] == "query: "

    def test_bge_profile(self):
        p = _get_profile("BAAI/bge-small-en-v1.5")
        assert p["prefix_query"].startswith("Represent")

    def test_jina_profile(self):
        p = _get_profile("jinaai/jina-embeddings-v5-text-nano")
        assert p["license"] == "CC-BY-NC-4.0"
        assert p["instruction_aware"] is True

    def test_nomic_matryoshka(self):
        p = _get_profile("nomic-ai/nomic-embed-text-v1.5")
        assert p["matryoshka"] is True
        assert p["max_tokens"] == 8192
