# -*- coding: utf-8 -*-
"""
██████╗  █████╗ ████████╗ █████╗
██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗
██║  ██║███████║   ██║   ███████║
██║  ██║██╔══██║   ██║   ██╔══██║
██████╔╝██║  ██║   ██║   ██║  ██║
╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝

Dataset loading, format conversion, and deduplication for LLM training data.
"""

from .dedup import DedupResult, TextDeduplicator, TextFilter
from .formatting import Conversation, DatasetFormatter
from .loader import load_dataset

__all__ = [
    "load_dataset",
    "DatasetFormatter",
    "Conversation",
    "TextDeduplicator",
    "TextFilter",
    "DedupResult",
]
