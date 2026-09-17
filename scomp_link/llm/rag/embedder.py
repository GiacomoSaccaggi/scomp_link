# -*- coding: utf-8 -*-
"""
███████╗███╗   ███╗██████╗ ███████╗██████╗
██╔════╝████╗ ████║██╔══██╗██╔════╝██╔══██╗
█████╗  ██╔████╔██║██████╔╝█████╗  ██║  ██║
██╔══╝  ██║╚██╔╝██║██╔══██╗██╔══╝  ██║  ██║
███████╗██║ ╚═╝ ██║██████╔╝███████╗██████╔╝
╚══════╝╚═╝     ╚═╝╚═════╝ ╚══════╝╚═════╝

Local embedding models with auto-prefix for Qwen3, E5, BGE, Nomic, Jina families.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Model profiles: auto-configure prefix style, dimensions, and context length
_MODEL_PROFILES: dict[str, dict] = {
    # ── Qwen3 Embedding (2025) — Apache 2.0, 32K context, Matryoshka ──
    "Qwen/Qwen3-Embedding-0.6B": {
        "dim": 1024,
        "max_tokens": 32768,
        "prefix_query": "query: ",
        "prefix_passage": "passage: ",
        "instruction_aware": True,
        "matryoshka": True,
        "license": "Apache-2.0",
    },
    "Qwen/Qwen3-Embedding-4B": {
        "dim": 2560,
        "max_tokens": 32768,
        "prefix_query": "query: ",
        "prefix_passage": "passage: ",
        "instruction_aware": True,
        "matryoshka": True,
        "license": "Apache-2.0",
    },
    "Qwen/Qwen3-Embedding-8B": {
        "dim": 4096,
        "max_tokens": 32768,
        "prefix_query": "query: ",
        "prefix_passage": "passage: ",
        "instruction_aware": True,
        "matryoshka": True,
        "license": "Apache-2.0",
    },
    # ── E5 family (2023) — multilingual, proven baseline ──
    "intfloat/multilingual-e5-small": {
        "dim": 384,
        "max_tokens": 512,
        "prefix_query": "query: ",
        "prefix_passage": "passage: ",
        "instruction_aware": False,
        "matryoshka": False,
        "license": "MIT",
    },
    "intfloat/multilingual-e5-base": {
        "dim": 768,
        "max_tokens": 512,
        "prefix_query": "query: ",
        "prefix_passage": "passage: ",
        "instruction_aware": False,
        "matryoshka": False,
        "license": "MIT",
    },
    "intfloat/multilingual-e5-large": {
        "dim": 1024,
        "max_tokens": 512,
        "prefix_query": "query: ",
        "prefix_passage": "passage: ",
        "instruction_aware": False,
        "matryoshka": False,
        "license": "MIT",
    },
    # ── BGE family (2024) — strong English, instruction-aware ──
    "BAAI/bge-small-en-v1.5": {
        "dim": 384,
        "max_tokens": 512,
        "prefix_query": "Represent this sentence for searching relevant passages: ",
        "prefix_passage": "",
        "instruction_aware": False,
        "matryoshka": False,
        "license": "MIT",
    },
    "BAAI/bge-base-en-v1.5": {
        "dim": 768,
        "max_tokens": 512,
        "prefix_query": "Represent this sentence for searching relevant passages: ",
        "prefix_passage": "",
        "instruction_aware": False,
        "matryoshka": False,
        "license": "MIT",
    },
    # ── Nomic Embed (2024-2025) — long context, Matryoshka ──
    "nomic-ai/nomic-embed-text-v1.5": {
        "dim": 768,
        "max_tokens": 8192,
        "prefix_query": "search_query: ",
        "prefix_passage": "search_document: ",
        "instruction_aware": False,
        "matryoshka": True,
        "license": "Apache-2.0",
    },
    "nomic-ai/nomic-embed-text-v2-moe": {
        "dim": 768,
        "max_tokens": 512,
        "prefix_query": "search_query: ",
        "prefix_passage": "search_document: ",
        "instruction_aware": False,
        "matryoshka": True,
        "license": "Apache-2.0",
    },
    # ── Jina v5 (2026) — task adapters, CC BY-NC 4.0 (non-commercial) ──
    "jinaai/jina-embeddings-v5-text-nano": {
        "dim": 768,
        "max_tokens": 8192,
        "prefix_query": "",
        "prefix_passage": "",
        "instruction_aware": True,
        "matryoshka": False,
        "license": "CC-BY-NC-4.0",
    },
    "jinaai/jina-embeddings-v5-text-small": {
        "dim": 1024,
        "max_tokens": 32768,
        "prefix_query": "",
        "prefix_passage": "",
        "instruction_aware": True,
        "matryoshka": False,
        "license": "CC-BY-NC-4.0",
    },
}

DEFAULT_EMBED_MODEL = "Qwen/Qwen3-Embedding-0.6B"
FALLBACK_EMBED_MODEL = "intfloat/multilingual-e5-small"


def _require_sentence_transformers():
    try:
        import sentence_transformers

        return sentence_transformers
    except ImportError:
        raise ImportError("LocalEmbedder requires sentence-transformers. " "Install with: pip install scomp-link[llm]")


def _get_profile(model_name: str) -> dict:
    """Return the profile for a known model, or a safe default for unknown ones."""
    if model_name in _MODEL_PROFILES:
        return _MODEL_PROFILES[model_name]
    # Check partial matches (user passes "Qwen3-Embedding-0.6B" without org prefix)
    for key, profile in _MODEL_PROFILES.items():
        if model_name in key or key.endswith(model_name):
            return profile
    # Unknown model: no prefix, passthrough
    return {
        "dim": None,
        "max_tokens": 512,
        "prefix_query": "",
        "prefix_passage": "",
        "instruction_aware": False,
        "matryoshka": False,
        "license": "unknown",
    }


_local_embedder: "LocalEmbedder | None" = None


def get_local_embedder(model_name: str | None = None) -> "LocalEmbedder":
    """Get or create a singleton LocalEmbedder."""
    global _local_embedder
    if model_name is None:
        if _local_embedder is not None:
            return _local_embedder
        model_name = DEFAULT_EMBED_MODEL
    if _local_embedder is None or _local_embedder._model_name != model_name:
        try:
            _local_embedder = LocalEmbedder(model_name)
        except Exception:
            if model_name != FALLBACK_EMBED_MODEL:
                logger.warning(
                    "Failed to load %s, falling back to %s",
                    model_name,
                    FALLBACK_EMBED_MODEL,
                )
                _local_embedder = LocalEmbedder(FALLBACK_EMBED_MODEL)
            else:
                raise
    return _local_embedder


class LocalEmbedder:
    """Multi-backend embedding with auto-configuration for known model families."""

    def __init__(self, model_name: str = DEFAULT_EMBED_MODEL):
        st = _require_sentence_transformers()
        self._model_name = model_name
        self._profile = _get_profile(model_name)
        self._model = st.SentenceTransformer(model_name, trust_remote_code=True)
        logger.info(
            "Loaded embedding model: %s (dim=%s, ctx=%s, license=%s)",
            model_name,
            self._profile["dim"],
            self._profile["max_tokens"],
            self._profile["license"],
        )

    @property
    def dim(self) -> int | None:
        return self._profile["dim"]

    @property
    def max_tokens(self) -> int:
        return self._profile["max_tokens"]

    @property
    def license(self) -> str:
        return self._profile["license"]

    def embed(
        self,
        texts: list[str],
        truncate_dim: int | None = None,
        instruction: str | None = None,
    ) -> list[list[float]]:
        prefix = self._profile["prefix_passage"]
        if instruction and self._profile["instruction_aware"]:
            prefixed = [f"{instruction}: {t}" for t in texts]
        elif prefix:
            prefixed = [f"{prefix}{t}" for t in texts]
        else:
            prefixed = texts

        embeddings = self._model.encode(prefixed, normalize_embeddings=True)
        result = embeddings.tolist()  # type: ignore[union-attr]

        if truncate_dim and self._profile["matryoshka"]:
            result = [e[:truncate_dim] for e in result]

        return result

    def embed_query(
        self,
        text: str,
        truncate_dim: int | None = None,
        instruction: str | None = None,
    ) -> list[float]:
        prefix = self._profile["prefix_query"]
        if instruction and self._profile["instruction_aware"]:
            formatted = f"{instruction}: {text}"
        elif prefix:
            formatted = f"{prefix}{text}"
        else:
            formatted = text

        embedding = self._model.encode(formatted, normalize_embeddings=True)
        result = embedding.tolist()  # type: ignore[union-attr]

        if truncate_dim and self._profile["matryoshka"]:
            result = result[:truncate_dim]

        return result

    @staticmethod
    def list_models() -> list[dict]:
        """Return info about all known embedding model profiles."""
        return [{"model": name, **{k: v for k, v in profile.items()}} for name, profile in _MODEL_PROFILES.items()]


if __name__ == "__main__":
    # Show all supported models and their specs
    print(f"{'Model':<45s} {'Dim':>5s} {'Ctx':>6s} {'Matr':>5s} {'License'}")
    print("-" * 80)
    for m in LocalEmbedder.list_models():
        dim = str(m["dim"] or "?")
        mat = "yes" if m["matryoshka"] else "no"
        print(f"{m['model']:<45s} {dim:>5s} {m['max_tokens']:>6d} {mat:>5s} {m['license']}")

    print(f"\nDefault: {DEFAULT_EMBED_MODEL}")
    print(f"Fallback: {FALLBACK_EMBED_MODEL}")

    # Test profile lookup
    p = _get_profile("Qwen/Qwen3-Embedding-0.6B")
    print(f"\nQwen3-0.6B profile: dim={p['dim']}, instruction_aware={p['instruction_aware']}")
