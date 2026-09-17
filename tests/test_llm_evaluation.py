# -*- coding: utf-8 -*-
"""Tests for scomp_link/llm/evaluation/ — ngrams, bleu, rouge, quality."""

import math

import pytest

from scomp_link.exceptions import DataValidationError
from scomp_link.llm.evaluation.bleu import BLEUScore
from scomp_link.llm.evaluation.ngrams import NGramAnalyzer
from scomp_link.llm.evaluation.quality import TextQualityMetrics
from scomp_link.llm.evaluation.rouge import ROUGEScore

# ── NGramAnalyzer ────────────────────────────────────────────────────────────


class TestNGramAnalyzer:
    def test_extract_unigrams(self):
        assert NGramAnalyzer.extract_ngrams("a b c", 1) == [("a",), ("b",), ("c",)]

    def test_extract_bigrams(self):
        bigrams = NGramAnalyzer.extract_ngrams("the cat sat on the mat", 2)
        assert len(bigrams) == 5

    def test_extract_trigrams(self):
        trigrams = NGramAnalyzer.extract_ngrams("a b c d e", 3)
        assert len(trigrams) == 3

    def test_extract_empty(self):
        assert NGramAnalyzer.extract_ngrams("", 1) == []

    def test_extract_n_larger_than_text(self):
        assert NGramAnalyzer.extract_ngrams("a b c", 5) == []

    def test_ngram_frequency(self):
        freq = NGramAnalyzer.ngram_frequency("a b a b a", 1)
        assert freq[("a",)] == 3
        assert freq[("b",)] == 2

    def test_diversity_all_unique(self):
        assert NGramAnalyzer.ngram_diversity("a b c d e", 1) == pytest.approx(1.0)

    def test_diversity_all_same(self):
        div = NGramAnalyzer.ngram_diversity("a a a a", 1)
        assert div == pytest.approx(1 / 4)

    def test_repetition_rate_no_repeat(self):
        rate = NGramAnalyzer.repetition_rate("a b c d e f g h", n=3)
        assert rate == pytest.approx(0.0)

    def test_repetition_rate_high(self):
        rate = NGramAnalyzer.repetition_rate("the cat the cat the cat", n=2)
        assert rate > 0.5

    def test_top_ngrams(self):
        top = NGramAnalyzer.top_ngrams("a b a b a b c", 1, k=2)
        assert top[0][0] == ("a",)
        assert top[0][1] >= top[1][1]


# ── BLEUScore ────────────────────────────────────────────────────────────────


class TestBLEUScore:
    def test_perfect_match(self):
        score = BLEUScore.sentence_bleu("hello world foo bar", "hello world foo bar")
        assert score == pytest.approx(1.0, abs=0.05)

    def test_partial_match(self):
        score = BLEUScore.sentence_bleu("the cat is on the mat", "the cat sat on the mat")
        assert 0.0 < score < 1.0

    def test_no_match(self):
        # add-1 smoothing inflates sentence BLEU; just verify it's well below partial
        score = BLEUScore.sentence_bleu("alpha beta gamma delta", "one two three four")
        assert score < 0.5

    def test_empty_hypothesis(self):
        assert BLEUScore.sentence_bleu("hello world", "") == 0.0

    def test_empty_reference(self):
        assert BLEUScore.sentence_bleu("", "hello world") == 0.0

    def test_brevity_penalty(self):
        long_score = BLEUScore.sentence_bleu("the big cat sat on the old mat", "the cat sat on the mat")
        short_score = BLEUScore.sentence_bleu("the big cat sat on the old mat", "the cat")
        assert long_score > short_score

    def test_corpus_bleu(self):
        # corpus BLEU needs enough n-gram overlap; use longer, more similar pairs
        refs = [
            "the cat sat on the mat in the room",
            "the dog ran across the park today",
        ]
        hyps = [
            "the cat sat on the mat in a room",
            "the dog ran across the large park today",
        ]
        score = BLEUScore.corpus_bleu(refs, hyps)
        assert 0.0 < score < 1.0

    def test_corpus_bleu_length_mismatch(self):
        with pytest.raises(DataValidationError, match="equal length"):
            BLEUScore.corpus_bleu(["a b"], ["a b", "c d"])


# ── ROUGEScore ───────────────────────────────────────────────────────────────


class TestROUGEScore:
    def test_rouge_1_perfect(self):
        r = ROUGEScore.rouge_n("the cat sat", "the cat sat", n=1)
        assert r["precision"] == pytest.approx(1.0)
        assert r["recall"] == pytest.approx(1.0)
        assert r["f1"] == pytest.approx(1.0)

    def test_rouge_1_partial(self):
        r = ROUGEScore.rouge_n("the cat sat on the mat", "the cat is here", n=1)
        assert 0.0 < r["f1"] < 1.0

    def test_rouge_2(self):
        r = ROUGEScore.rouge_n("the cat sat on the mat", "the cat sat on a mat", n=2)
        assert 0.0 < r["f1"] < 1.0

    def test_rouge_l_perfect(self):
        r = ROUGEScore.rouge_l("a b c d", "a b c d")
        assert r["f1"] == pytest.approx(1.0)

    def test_rouge_l_partial(self):
        r = ROUGEScore.rouge_l("a b c d e", "a c e")
        assert 0.0 < r["f1"] < 1.0

    def test_rouge_empty(self):
        for fn in [
            lambda: ROUGEScore.rouge_n("", "hello", n=1),
            lambda: ROUGEScore.rouge_n("hello", "", n=1),
            lambda: ROUGEScore.rouge_l("", "hello"),
            lambda: ROUGEScore.rouge_l("hello", ""),
        ]:
            r = fn()
            assert r["precision"] == 0.0
            assert r["recall"] == 0.0
            assert r["f1"] == 0.0


# ── TextQualityMetrics ───────────────────────────────────────────────────────


class TestTextQualityMetrics:
    def test_evaluate_with_reference(self):
        m = TextQualityMetrics.evaluate("the cat sat on the mat", references="the cat is on the mat")
        assert "bleu" in m
        assert "rouge_1_f1" in m
        assert "rouge_l_f1" in m
        assert "diversity_1" in m

    def test_evaluate_without_reference(self):
        m = TextQualityMetrics.evaluate("the cat sat on the mat and looked around")
        assert "bleu" not in m
        assert "diversity_1" in m
        assert "repetition_rate" in m
        assert "zipf_coefficient" in m

    def test_self_bleu_identical(self):
        texts = ["the cat sat on the mat"] * 5
        sb = TextQualityMetrics.self_bleu(texts)
        assert sb > 0.8

    def test_self_bleu_diverse(self):
        texts = [
            "cats like fish very much indeed",
            "dogs prefer bones and toys",
            "birds enjoy seeds and worms",
        ]
        sb = TextQualityMetrics.self_bleu(texts)
        assert sb < 0.3

    def test_perplexity_from_loss(self):
        ppl = TextQualityMetrics.perplexity_from_loss(2.0)
        assert ppl == pytest.approx(math.exp(2.0), rel=1e-6)


class TestNGramEdgeCases:
    def test_extract_invalid_n(self):
        with pytest.raises(DataValidationError):
            NGramAnalyzer.extract_ngrams("hello", 0)

    def test_diversity_empty(self):
        assert NGramAnalyzer.ngram_diversity("", 1) == 0.0

    def test_repetition_rate_empty(self):
        assert NGramAnalyzer.repetition_rate("", 3) == 0.0
