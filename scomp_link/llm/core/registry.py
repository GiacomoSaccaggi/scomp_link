# -*- coding: utf-8 -*-
"""
██████╗ ███████╗ ██████╗ ██╗███████╗████████╗██████╗ ██╗   ██╗
██╔══██╗██╔════╝██╔════╝ ██║██╔════╝╚══██╔══╝██╔══██╗╚██╗ ██╔╝
██████╔╝█████╗  ██║  ███╗██║███████╗   ██║   ██████╔╝ ╚████╔╝
██╔══██╗██╔══╝  ██║   ██║██║╚════██║   ██║   ██╔══██╗  ╚██╔╝
██║  ██║███████╗╚██████╔╝██║███████║   ██║   ██║  ██║   ██║
╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═╝╚══════╝   ╚═╝   ╚═╝  ╚═╝   ╚═╝

Lazy capability registry — maps short names to (module, class) pairs.
"""

import importlib

# Each entry: "capability_name" → (".subpackage.module", "ClassName")
# Paths are relative to scomp_link.llm
_CAPABILITIES: dict[str, tuple[str, str]] = {
    "scratch": (".training.scratch", "TransformerBuilder"),
    "finetune": (".training.finetune", "FineTuner"),
    "convert": (".serving.convert", "ModelConverter"),
    "rag": (".rag.pipeline", "RAGPipeline"),
    "evaluate": (".evaluation.quality", "TextQualityMetrics"),
    "format": (".data.formatting", "DatasetFormatter"),
    "dedup": (".data.dedup", "TextDeduplicator"),
    "merge": (".serving.merge", "ModelMerger"),
    "serve": (".serving.serve", "InferenceServer"),
}


def get_capability(name: str) -> type:
    module_path, class_name = _CAPABILITIES[name]
    module = importlib.import_module(module_path, package="scomp_link.llm")
    return getattr(module, class_name)


if __name__ == "__main__":
    # Print the registry map and resolve one that's cheap to import
    for cap, (mod, cls) in sorted(_CAPABILITIES.items()):
        print(f"  {cap:12s} → {mod}::{cls}")

    # Actually resolve 'evaluate' — it's pure Python, no heavy deps
    klass = get_capability("evaluate")
    print(f"\nResolved 'evaluate' → {klass.__name__}")
