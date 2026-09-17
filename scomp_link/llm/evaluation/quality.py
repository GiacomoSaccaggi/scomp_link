# -*- coding: utf-8 -*-
"""
 ██████╗ ██╗   ██╗ █████╗ ██╗     ██╗████████╗██╗   ██╗
██╔═══██╗██║   ██║██╔══██╗██║     ██║╚══██╔══╝╚██╗ ██╔╝
██║   ██║██║   ██║███████║██║     ██║   ██║    ╚████╔╝
██║▄▄ ██║██║   ██║██╔══██║██║     ██║   ██║     ╚██╔╝
╚██████╔╝╚██████╔╝██║  ██║███████╗██║   ██║      ██║
 ╚══▀▀═╝  ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝   ╚═╝      ╚═╝

Aggregate text quality metrics: BLEU, ROUGE, diversity, Zipf, perplexity, self-BLEU.
"""

from __future__ import annotations

import math
import random
from collections import Counter

from scomp_link.exceptions import DataValidationError
from scomp_link.llm.evaluation.bleu import BLEUScore
from scomp_link.llm.evaluation.ngrams import NGramAnalyzer
from scomp_link.llm.evaluation.rouge import ROUGEScore


def _zipf_coefficient(tokens: list[str]) -> float:
    """Fit Zipf's law via least-squares on log-log rank vs frequency."""
    freq = Counter(tokens)
    if len(freq) < 2:
        return 0.0
    counts = sorted(freq.values(), reverse=True)
    n = len(counts)
    log_ranks = [math.log(i + 1) for i in range(n)]
    log_freqs = [math.log(c) for c in counts]
    mean_x = sum(log_ranks) / n
    mean_y = sum(log_freqs) / n
    num = sum((log_ranks[i] - mean_x) * (log_freqs[i] - mean_y) for i in range(n))
    den = sum((log_ranks[i] - mean_x) ** 2 for i in range(n))
    if den == 0.0:
        return 0.0
    return -num / den  # Negate: Zipf exponent is positive


class TextQualityMetrics:
    """Aggregate text quality assessment."""

    @staticmethod
    def evaluate(
        generated: str | list[str],
        references: str | list[str] | None = None,
    ) -> dict:
        if isinstance(generated, str):
            generated = [generated]
        if isinstance(references, str):
            references = [references]

        combined = " ".join(generated)
        tokens = NGramAnalyzer._tokenize(combined)
        sentences = [s.strip() for s in combined.replace("!", ".").replace("?", ".").split(".") if s.strip()]

        metrics: dict[str, float] = {}

        # Reference-based metrics
        if references is not None:
            if len(references) != len(generated):
                raise DataValidationError(
                    f"generated and references must have equal length, " f"got {len(generated)} vs {len(references)}"
                )
            metrics["bleu"] = BLEUScore.corpus_bleu(references, generated)
            avg_r1 = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            avg_r2 = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            avg_rl = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
            for ref, hyp in zip(references, generated):
                for d, fn, n in [
                    (avg_r1, ROUGEScore.rouge_n, 1),
                    (avg_r2, ROUGEScore.rouge_n, 2),
                ]:
                    r = fn(ref, hyp, n=n)
                    for k in d:
                        d[k] += r[k]
                rl = ROUGEScore.rouge_l(ref, hyp)
                for k in avg_rl:
                    avg_rl[k] += rl[k]
            n_pairs = len(references)
            for key_prefix, d in [
                ("rouge_1", avg_r1),
                ("rouge_2", avg_r2),
                ("rouge_l", avg_rl),
            ]:
                for k, v in d.items():
                    metrics[f"{key_prefix}_{k}"] = v / n_pairs

        # Intrinsic metrics
        for n in (1, 2, 3, 4):
            metrics[f"diversity_{n}"] = NGramAnalyzer.ngram_diversity(combined, n)
        metrics["repetition_rate"] = NGramAnalyzer.repetition_rate(combined)
        metrics["avg_sentence_length"] = sum(len(s.split()) for s in sentences) / len(sentences) if sentences else 0.0
        metrics["vocabulary_richness"] = len(set(tokens)) / len(tokens) if tokens else 0.0
        metrics["zipf_coefficient"] = _zipf_coefficient(tokens)

        return metrics

    @staticmethod
    def perplexity_from_loss(avg_loss: float) -> float:
        return math.exp(avg_loss)

    @staticmethod
    def self_bleu(texts: list[str], sample_size: int = 100) -> float:
        if len(texts) < 2:
            return 0.0
        if len(texts) > sample_size:
            texts = random.sample(texts, sample_size)
        total = 0.0
        for i, hyp in enumerate(texts):
            others = texts[:i] + texts[i + 1 :]
            bleu_scores = [BLEUScore.sentence_bleu(ref, hyp) for ref in others]
            total += sum(bleu_scores) / len(bleu_scores)
        return total / len(texts)


if __name__ == "__main__":
    # Evaluate with reference
    gen = "The cat sat on the mat and looked out the window"
    ref = "The cat is sitting on the mat near the window"
    metrics = TextQualityMetrics.evaluate(gen, references=ref)
    print("With reference:")
    for k in [
        "bleu",
        "rouge_1_f1",
        "rouge_l_f1",
        "diversity_1",
        "diversity_2",
        "vocabulary_richness",
        "zipf_coefficient",
    ]:
        print(f"  {k}: {metrics[k]:.4f}")

    # Self-BLEU: how diverse are these 3 texts?
    texts = ["cats like fish", "dogs like bones", "birds like seeds"]
    sb = TextQualityMetrics.self_bleu(texts)
    print(f"\nSelf-BLEU (lower=more diverse): {sb:.4f}")

    # Perplexity from loss
    print(f"PPL from loss=2.3: {TextQualityMetrics.perplexity_from_loss(2.3):.2f}")
