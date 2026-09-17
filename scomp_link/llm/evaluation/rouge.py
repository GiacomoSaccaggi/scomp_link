# -*- coding: utf-8 -*-
"""
██████╗  ██████╗ ██╗   ██╗ ██████╗ ███████╗
██╔══██╗██╔═══██╗██║   ██║██╔════╝ ██╔════╝
██████╔╝██║   ██║██║   ██║██║  ███╗█████╗
██╔══██╗██║   ██║██║   ██║██║   ██║██╔══╝
██║  ██║╚██████╔╝╚██████╔╝╚██████╔╝███████╗
╚═╝  ╚═╝ ╚═════╝  ╚═════╝  ╚═════╝ ╚══════╝

ROUGE scores: n-gram recall (ROUGE-N) and longest common subsequence (ROUGE-L).
"""

from __future__ import annotations

from collections import Counter

from scomp_link.llm.evaluation.ngrams import NGramAnalyzer


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


class ROUGEScore:
    """ROUGE (Recall-Oriented Understudy for Gisting Evaluation).

    ROUGE-N: n-gram recall between reference and hypothesis.
    ROUGE-L: longest common subsequence based F-score.
    """

    @staticmethod
    def rouge_n(reference: str, hypothesis: str, n: int = 1) -> dict[str, float]:
        ref_ngrams = Counter(NGramAnalyzer.extract_ngrams(reference, n))
        hyp_ngrams = Counter(NGramAnalyzer.extract_ngrams(hypothesis, n))
        ref_total = sum(ref_ngrams.values())
        hyp_total = sum(hyp_ngrams.values())
        if ref_total == 0 or hyp_total == 0:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        overlap = 0
        for ng, count in hyp_ngrams.items():
            overlap += min(count, ref_ngrams.get(ng, 0))
        precision = overlap / hyp_total
        recall = overlap / ref_total
        return {"precision": precision, "recall": recall, "f1": _f1(precision, recall)}

    @staticmethod
    def _lcs_length(x: list[str], y: list[str]) -> int:
        m, n = len(x), len(y)
        # Space-optimized DP: two rows
        prev = [0] * (n + 1)
        curr = [0] * (n + 1)
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if x[i - 1] == y[j - 1]:
                    curr[j] = prev[j - 1] + 1
                else:
                    curr[j] = max(prev[j], curr[j - 1])
            prev, curr = curr, [0] * (n + 1)
        return prev[n] if m > 0 else 0

    @staticmethod
    def rouge_l(reference: str, hypothesis: str) -> dict[str, float]:
        ref_tokens = NGramAnalyzer._tokenize(reference)
        hyp_tokens = NGramAnalyzer._tokenize(hypothesis)
        if not ref_tokens or not hyp_tokens:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        lcs = ROUGEScore._lcs_length(ref_tokens, hyp_tokens)
        precision = lcs / len(hyp_tokens)
        recall = lcs / len(ref_tokens)
        return {"precision": precision, "recall": recall, "f1": _f1(precision, recall)}


if __name__ == "__main__":
    ref = "the cat is sitting on the mat near the window"
    hyp = "a cat sits on the mat"

    for n in (1, 2):
        r = ROUGEScore.rouge_n(ref, hyp, n=n)
        print(f"ROUGE-{n}: P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f}")

    rl = ROUGEScore.rouge_l(ref, hyp)
    print(f"ROUGE-L: P={rl['precision']:.3f} R={rl['recall']:.3f} F1={rl['f1']:.3f}")
