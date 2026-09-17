# -*- coding: utf-8 -*-
"""
scomp_link.llm — LLM Toolkit for scomp-link.

Fine-tune pretrained models (LoRA, QLoRA, full), build transformers from
scratch, convert to GGUF, and more.  Heavy dependencies (torch, transformers,
peft, …) are optional — install with ``pip install scomp-link[llm]``.

All public names are lazily imported via PEP 562 (__getattr__), so
``import scomp_link.llm`` is near-instant regardless of which extras
are installed.
"""

__all__ = [
    # Core classes
    "FineTuner",
    "TransformerBuilder",
    "ModelConverter",
    "RAGPipeline",
    # Factory & registry
    "LLMFactory",
    # Configs
    "TransformerConfig",
    "FineTuneConfig",
    # Results
    "TrainResult",
    "ConvertResult",
    # Evaluation
    "NGramAnalyzer",
    "BLEUScore",
    "ROUGEScore",
    "TextQualityMetrics",
    # Formatting
    "DatasetFormatter",
    "Conversation",
    # Dedup
    "TextDeduplicator",
    "TextFilter",
    "DedupResult",
    # Merge
    "ModelMerger",
    # Serve
    "InferenceServer",
    # RAG
    "Chunk",
    "RetrievalResult",
]

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    # Capabilities
    "FineTuner": (".training.finetune", "FineTuner"),
    "TransformerBuilder": (".training.scratch", "TransformerBuilder"),
    "ModelConverter": (".serving.convert", "ModelConverter"),
    "RAGPipeline": (".rag.pipeline", "RAGPipeline"),
    # Factory
    "LLMFactory": (".core.factory", "LLMFactory"),
    # Configs & results
    "TransformerConfig": (".core.configs", "TransformerConfig"),
    "FineTuneConfig": (".core.configs", "FineTuneConfig"),
    "TrainResult": (".core.configs", "TrainResult"),
    "ConvertResult": (".core.configs", "ConvertResult"),
    # Evaluation
    "NGramAnalyzer": (".evaluation.ngrams", "NGramAnalyzer"),
    "BLEUScore": (".evaluation.bleu", "BLEUScore"),
    "ROUGEScore": (".evaluation.rouge", "ROUGEScore"),
    "TextQualityMetrics": (".evaluation.quality", "TextQualityMetrics"),
    # Formatting
    "DatasetFormatter": (".data.formatting", "DatasetFormatter"),
    "Conversation": (".data.formatting", "Conversation"),
    # Dedup
    "TextDeduplicator": (".data.dedup", "TextDeduplicator"),
    "TextFilter": (".data.dedup", "TextFilter"),
    "DedupResult": (".data.dedup", "DedupResult"),
    # Merge
    "ModelMerger": (".serving.merge", "ModelMerger"),
    # Serve
    "InferenceServer": (".serving.serve", "InferenceServer"),
    # RAG
    "Chunk": (".rag.chunking", "Chunk"),
    "RetrievalResult": (".rag.pipeline", "RetrievalResult"),
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        module_path, attr_name = _LAZY_IMPORTS[name]
        import importlib

        module = importlib.import_module(module_path, __package__)
        obj = getattr(module, attr_name)
        globals()[name] = obj
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return list(__all__)
