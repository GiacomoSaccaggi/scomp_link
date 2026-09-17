# -*- coding: utf-8 -*-
"""
███████╗██╗   ██╗ █████╗ ██╗
██╔════╝██║   ██║██╔══██╗██║
█████╗  ██║   ██║███████║██║
██╔══╝  ╚██╗ ██╔╝██╔══██║██║
███████╗ ╚████╔╝ ██║  ██║███████╗
╚══════╝  ╚═══╝  ╚═╝  ╚═╝╚══════╝

Text evaluation: BLEU, ROUGE, n-gram analysis, and aggregate quality metrics.
"""

from .bleu import BLEUScore
from .ngrams import NGramAnalyzer
from .quality import TextQualityMetrics
from .rouge import ROUGEScore

__all__ = [
    "NGramAnalyzer",
    "BLEUScore",
    "ROUGEScore",
    "TextQualityMetrics",
]
