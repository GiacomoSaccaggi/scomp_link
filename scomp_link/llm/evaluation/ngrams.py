# -*- coding: utf-8 -*-
"""
███╗   ██╗ ██████╗ ██████╗  █████╗ ███╗   ███╗███████╗
████╗  ██║██╔════╝ ██╔══██╗██╔══██╗████╗ ████║██╔════╝
██╔██╗ ██║██║  ███╗██████╔╝███████║██╔████╔██║███████╗
██║╚██╗██║██║   ██║██╔══██╗██╔══██║██║╚██╔╝██║╚════██║
██║ ╚████║╚██████╔╝██║  ██║██║  ██║██║ ╚═╝ ██║███████║
╚═╝  ╚═══╝ ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝

Core n-gram computation: extraction, frequency, diversity, repetition.
"""

from __future__ import annotations

from collections import Counter

from scomp_link.exceptions import DataValidationError


class NGramAnalyzer:
    """Core n-gram computation engine."""

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()

    @staticmethod
    def extract_ngrams(text: str, n: int) -> list[tuple[str, ...]]:
        if n < 1:
            raise DataValidationError(f"n must be >= 1, got {n}")
        tokens = NGramAnalyzer._tokenize(text)
        if len(tokens) < n:
            return []
        return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]

    @staticmethod
    def ngram_frequency(text: str, n: int) -> dict[tuple[str, ...], int]:
        return dict(Counter(NGramAnalyzer.extract_ngrams(text, n)))

    @staticmethod
    def ngram_diversity(text: str, n: int) -> float:
        ngrams = NGramAnalyzer.extract_ngrams(text, n)
        if not ngrams:
            return 0.0
        return len(set(ngrams)) / len(ngrams)

    @staticmethod
    def repetition_rate(text: str, n: int = 3) -> float:
        freq = Counter(NGramAnalyzer.extract_ngrams(text, n))
        total = sum(freq.values())
        if total == 0:
            return 0.0
        repeated = sum(c for c in freq.values() if c > 1)
        return repeated / total

    @staticmethod
    def top_ngrams(text: str, n: int, k: int = 20) -> list[tuple[tuple[str, ...], int]]:
        freq = Counter(NGramAnalyzer.extract_ngrams(text, n))
        return freq.most_common(k)


if __name__ == "__main__":
    text = "the quick brown fox jumps over the lazy brown dog"

    # Extract and count
    bigrams = NGramAnalyzer.extract_ngrams(text, 2)
    print(f"Bigrams ({len(bigrams)}): {bigrams[:5]}...")

    # Diversity: unique/total ratio
    for n in (1, 2, 3):
        d = NGramAnalyzer.ngram_diversity(text, n)
        print(f"  {n}-gram diversity: {d:.3f}")

    # Most frequent
    print(f"\nTop trigrams: {NGramAnalyzer.top_ngrams(text, 3, k=3)}")

    # Repetition rate
    rep = NGramAnalyzer.repetition_rate("the cat the cat the cat", n=2)
    print(f"Repetition rate of 'the cat the cat the cat': {rep:.2%}")
