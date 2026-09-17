# -*- coding: utf-8 -*-
"""
██████╗ ██╗██████╗ ███████╗██╗     ██╗███╗   ██╗███████╗
██╔══██╗██║██╔══██╗██╔════╝██║     ██║████╗  ██║██╔════╝
██████╔╝██║██████╔╝█████╗  ██║     ██║██╔██╗ ██║█████╗
██╔═══╝ ██║██╔═══╝ ██╔══╝  ██║     ██║██║╚██╗██║██╔══╝
██║     ██║██║     ███████╗███████╗██║██║ ╚████║███████╗
╚═╝     ╚═╝╚═╝     ╚══════╝╚══════╝╚═╝╚═╝  ╚═══╝╚══════╝

RAG pipeline: build/query ChromaDB indexes with hybrid (dense + keyword) retrieval.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

from scomp_link.llm.rag.chunking import Chunk, chunk_directory
from scomp_link.llm.rag.embedder import get_local_embedder
from scomp_link.llm.rag.guardrails import (
    build_rag_prompt,
    format_provenance,
    validate_query,
    validate_response,
)

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    text: str
    score: float
    source_file: str
    chunk_type: str
    start_line: int
    end_line: int


def _require_chromadb():
    try:
        import chromadb

        return chromadb
    except ImportError:
        raise ImportError("RAGPipeline requires chromadb. Install with: pip install scomp-link[llm]")


def _sanitize_collection_name(name: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9._-]", "_", name)
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    if len(sanitized) < 3:
        sanitized = sanitized + "_" * (3 - len(sanitized))
    return f"rag_{sanitized[:500]}"


class RAGPipeline:
    def __init__(self, persist_dir: str = "./rag_data", embed_fn: Any = None):
        chromadb = _require_chromadb()
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._persist_dir = persist_dir
        if embed_fn is not None:
            self._embedder = embed_fn
        else:
            self._embedder = get_local_embedder()

    def build_index(self, name: str, path: str, extra_texts: list[Chunk] | None = None) -> dict:
        chunks = chunk_directory(path)
        if extra_texts:
            chunks.extend(extra_texts)
        if not chunks:
            return {"status": "ok", "chunks": 0, "files": 0}

        col_name = _sanitize_collection_name(name)
        collection = self._client.get_or_create_collection(name=col_name)

        batch_size = 32
        files: set[str] = set()
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            texts = [c.text for c in batch]
            embeddings = self._embedder.embed(texts)
            ids = [f"{col_name}_{i + j}" for j in range(len(batch))]
            metadatas = [
                {
                    "source_file": c.source_file,
                    "chunk_type": c.chunk_type,
                    "start_line": c.start_line,
                    "end_line": c.end_line,
                    **c.metadata,
                }
                for c in batch
            ]
            collection.upsert(
                ids=ids,
                embeddings=embeddings,  # type: ignore[arg-type]
                documents=texts,
                metadatas=metadatas,  # type: ignore[arg-type]
            )
            files.update(c.source_file for c in batch)

        return {"status": "ok", "chunks": len(chunks), "files": len(files)}

    def query(self, name: str, question: str, top_k: int = 5) -> list[RetrievalResult]:
        col_name = _sanitize_collection_name(name)
        collection = self._client.get_collection(name=col_name)
        q_embedding = self._embedder.embed_query(question)
        results = collection.query(query_embeddings=[q_embedding], n_results=top_k)

        docs = results["documents"] or [[]]
        metas = results["metadatas"] or [[]]
        dists = results["distances"] or [[]]

        items: list[RetrievalResult] = []
        for doc, meta, dist in zip(docs[0], metas[0], dists[0]):
            score = 1.0 - dist
            items.append(
                RetrievalResult(
                    text=doc,
                    score=score,
                    source_file=str(meta.get("source_file", "")),
                    chunk_type=str(meta.get("chunk_type", "paragraph")),
                    start_line=int(meta.get("start_line", 0)),
                    end_line=int(meta.get("end_line", 0)),
                )
            )
        return items

    def hybrid_query(self, name: str, question: str, top_k: int = 10) -> list[RetrievalResult]:
        dense_results = self.query(name, question, top_k=top_k)

        col_name = _sanitize_collection_name(name)
        collection = self._client.get_collection(name=col_name)
        keywords = [w for w in question.lower().split() if len(w) > 2][:3]
        keyword_results: list[RetrievalResult] = []
        for kw in keywords:
            try:
                kw_res = collection.query(
                    query_embeddings=[self._embedder.embed_query(kw)],
                    n_results=top_k // 2,
                    where_document={"$contains": kw},
                )
                kw_docs = kw_res["documents"] or [[]]
                kw_metas = kw_res["metadatas"] or [[]]
                kw_dists = kw_res["distances"] or [[]]
                for doc, meta, dist in zip(kw_docs[0], kw_metas[0], kw_dists[0]):
                    keyword_results.append(
                        RetrievalResult(
                            text=doc,
                            score=(1.0 - dist) * 0.8,
                            source_file=str(meta.get("source_file", "")),
                            chunk_type=str(meta.get("chunk_type", "paragraph")),
                            start_line=int(meta.get("start_line", 0)),
                            end_line=int(meta.get("end_line", 0)),
                        )
                    )
            except Exception:
                pass

        seen: set[str] = set()
        merged: list[RetrievalResult] = []
        for r in sorted(dense_results + keyword_results, key=lambda x: x.score, reverse=True):
            key = r.text[:200]
            if key not in seen:
                seen.add(key)
                merged.append(r)
        return merged[:top_k]

    def delete_index(self, name: str) -> None:
        col_name = _sanitize_collection_name(name)
        self._client.delete_collection(name=col_name)

    def has_index(self, name: str) -> bool:
        col_name = _sanitize_collection_name(name)
        try:
            self._client.get_collection(name=col_name)
            return True
        except Exception:
            return False

    def index_status(self, name: str) -> dict:
        col_name = _sanitize_collection_name(name)
        try:
            collection = self._client.get_collection(name=col_name)
            return {"exists": True, "chunks": collection.count(), "collection": col_name}
        except Exception:
            return {"exists": False, "chunks": 0, "collection": col_name}

    # Convenience: attach guardrail functions as static methods
    validate_query = staticmethod(validate_query)
    validate_response = staticmethod(validate_response)
    build_prompt = staticmethod(build_rag_prompt)
    format_provenance = staticmethod(format_provenance)


if __name__ == "__main__":
    # Demo the collection name sanitizer
    names = ["my project!", "café résumé", "ab", "hello/world@2024", "a" * 600]
    print("Collection name sanitization:")
    for n in names:
        print(f"  {n[:30]!r:35s} → {_sanitize_collection_name(n)}")

    # Show RetrievalResult
    r = RetrievalResult(
        text="def hello(): pass",
        score=0.95,
        source_file="main.py",
        chunk_type="python_function",
        start_line=1,
        end_line=1,
    )
    print(f"\nRetrievalResult: score={r.score}, file={r.source_file}, type={r.chunk_type}")
