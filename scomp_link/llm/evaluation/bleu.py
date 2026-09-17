# -*- coding: utf-8 -*-
"""
██████╗ ██╗     ███████╗██╗   ██╗
██╔══██╗██║     ██╔════╝██║   ██║
██████╔╝██║     █████╗  ██║   ██║
██╔══██╗██║     ██╔══╝  ██║   ██║
██████╔╝███████╗███████╗╚██████╔╝
╚═════╝ ╚══════╝╚══════╝ ╚═════╝

BLEU score: n-gram precision with brevity penalty, sentence and corpus level.
"""

from __future__ import annotations

import math
from collections import Counter

from scomp_link.exceptions import DataValidationError
from scomp_link.llm.evaluation.ngrams import NGramAnalyzer


class BLEUScore:
    """BLEU (Bilingual Evaluation Understudy) — measures n-gram precision.

    BLEU = BP * exp(Σ wₙ * log(pₙ))
    where pₙ = clipped n-gram precision, BP = brevity penalty.
    """

    @staticmethod
    def _clipped_precision(ref_tokens: list[str], hyp_tokens: list[str], n: int) -> tuple[int, int]:
        ref_ngrams = Counter(tuple(ref_tokens[i : i + n]) for i in range(len(ref_tokens) - n + 1))
        hyp_ngrams = Counter(tuple(hyp_tokens[i : i + n]) for i in range(len(hyp_tokens) - n + 1))
        clipped = 0
        for ng, count in hyp_ngrams.items():
            clipped += min(count, ref_ngrams.get(ng, 0))
        total = max(sum(hyp_ngrams.values()), 0)
        return clipped, total

    @staticmethod
    def _brevity_penalty(ref_len: int, hyp_len: int) -> float:
        if hyp_len == 0:
            return 0.0
        if hyp_len >= ref_len:
            return 1.0
        return math.exp(1.0 - ref_len / hyp_len)

    @staticmethod
    def sentence_bleu(
        reference: str,
        hypothesis: str,
        max_n: int = 4,
        weights: tuple | None = None,
    ) -> float:
        ref_tokens = NGramAnalyzer._tokenize(reference)
        hyp_tokens = NGramAnalyzer._tokenize(hypothesis)
        if not hyp_tokens or not ref_tokens:
            return 0.0
        if weights is None:
            weights = tuple(1.0 / max_n for _ in range(max_n))
        bp = BLEUScore._brevity_penalty(len(ref_tokens), len(hyp_tokens))
        log_avg = 0.0
        for n_idx in range(max_n):
            n = n_idx + 1
            clipped, total = BLEUScore._clipped_precision(ref_tokens, hyp_tokens, n)
            # Add-1 smoothing for sentence-level BLEU
            precision = (clipped + 1) / (total + 1)
            log_avg += weights[n_idx] * math.log(precision)
        return bp * math.exp(log_avg)

    @staticmethod
    def corpus_bleu(
        references: list[str],
        hypotheses: list[str],
        max_n: int = 4,
    ) -> float:
        if len(references) != len(hypotheses):
            raise DataValidationError(
                f"references and hypotheses must have equal length, " f"got {len(references)} vs {len(hypotheses)}"
            )
        if not references:
            return 0.0
        total_clipped = [0] * max_n
        total_count = [0] * max_n
        ref_length = 0
        hyp_length = 0
        for ref, hyp in zip(references, hypotheses):
            ref_tokens = NGramAnalyzer._tokenize(ref)
            hyp_tokens = NGramAnalyzer._tokenize(hyp)
            ref_length += len(ref_tokens)
            hyp_length += len(hyp_tokens)
            for n_idx in range(max_n):
                n = n_idx + 1
                c, t = BLEUScore._clipped_precision(ref_tokens, hyp_tokens, n)
                total_clipped[n_idx] += c
                total_count[n_idx] += t
        bp = BLEUScore._brevity_penalty(ref_length, hyp_length)
        weights = tuple(1.0 / max_n for _ in range(max_n))
        log_avg = 0.0
        for n_idx in range(max_n):
            if total_count[n_idx] == 0:
                return 0.0
            precision = total_clipped[n_idx] / total_count[n_idx]
            if precision == 0.0:
                return 0.0
            log_avg += weights[n_idx] * math.log(precision)
        return bp * math.exp(log_avg)


if __name__ == "__main__":
    pairs = [
        ("the cat is on the mat", "the cat sat on the mat"),
        ("it is raining today", "today it rains"),
        ("hello world", "hello world"),  # perfect match
    ]
    for ref, hyp in pairs:
        score = BLEUScore.sentence_bleu(ref, hyp)
        print(f"  BLEU={score:.4f}  ref={ref!r}  hyp={hyp!r}")

    # Corpus BLEU
    refs = [p[0] for p in pairs]
    hyps = [p[1] for p in pairs]
    corpus = BLEUScore.corpus_bleu(refs, hyps)
    print(f"\nCorpus BLEU: {corpus:.4f}")
