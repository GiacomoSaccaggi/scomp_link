# -*- coding: utf-8 -*-
"""
██████╗  █████╗  ██████╗
██╔══██╗██╔══██╗██╔════╝
██████╔╝███████║██║  ███╗
██╔══██╗██╔══██║██║   ██║
██║  ██║██║  ██║╚██████╔╝
╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝

Retrieval-Augmented Generation: chunking, embedding, guardrails, and pipeline.
"""

from .chunking import (
    Chunk,
    chunk_directory,
    chunk_file,
    chunk_markdown,
    chunk_python,
    chunk_sql,
    chunk_text,
    chunk_yaml,
)
from .embedder import (
    DEFAULT_EMBED_MODEL,
    FALLBACK_EMBED_MODEL,
    LocalEmbedder,
    get_local_embedder,
)
from .guardrails import (
    INJECTION_PATTERNS,
    build_rag_prompt,
    format_provenance,
    validate_query,
    validate_response,
)
from .pipeline import RAGPipeline, RetrievalResult

__all__ = [
    "RAGPipeline",
    "Chunk",
    "RetrievalResult",
    "LocalEmbedder",
    "get_local_embedder",
    "chunk_python",
    "chunk_yaml",
    "chunk_markdown",
    "chunk_sql",
    "chunk_text",
    "chunk_file",
    "chunk_directory",
    "validate_query",
    "validate_response",
    "build_rag_prompt",
    "format_provenance",
    "INJECTION_PATTERNS",
    "DEFAULT_EMBED_MODEL",
    "FALLBACK_EMBED_MODEL",
]
