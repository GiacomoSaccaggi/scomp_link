# -*- coding: utf-8 -*-
"""
██████╗ ███████╗██████╗ ██╗   ██╗██████╗
██╔══██╗██╔════╝██╔══██╗██║   ██║██╔══██╗
██║  ██║█████╗  ██║  ██║██║   ██║██████╔╝
██║  ██║██╔══╝  ██║  ██║██║   ██║██╔═══╝
██████╔╝███████╗██████╔╝╚██████╔╝██║
╚═════╝ ╚══════╝╚═════╝  ╚═════╝ ╚═╝

Text deduplication (exact hash + MinHash LSH) and quality filtering for training data.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass

from scomp_link.exceptions import DataValidationError

_ENGLISH_STOP_WORDS = frozenset(
    {
        "the",
        "be",
        "to",
        "of",
        "and",
        "a",
        "in",
        "that",
        "have",
        "i",
        "it",
        "for",
        "not",
        "on",
        "with",
        "he",
        "as",
        "you",
        "do",
        "at",
        "this",
        "but",
        "his",
        "by",
        "from",
        "they",
        "we",
        "say",
        "her",
        "she",
        "or",
        "an",
        "will",
        "my",
        "one",
        "all",
        "would",
        "there",
        "their",
        "what",
        "so",
        "up",
        "out",
        "if",
        "about",
        "who",
        "get",
        "which",
        "go",
        "me",
        "when",
        "make",
        "can",
        "like",
        "time",
        "no",
        "just",
        "him",
        "know",
        "take",
        "people",
        "into",
        "year",
        "your",
        "good",
        "some",
        "could",
        "them",
        "see",
        "other",
        "than",
        "then",
        "now",
        "look",
        "only",
        "come",
        "its",
        "over",
        "think",
        "also",
        "back",
        "after",
        "use",
        "two",
        "how",
        "our",
        "work",
        "first",
        "well",
        "way",
        "even",
        "new",
        "want",
        "because",
        "any",
        "these",
        "give",
        "day",
        "most",
        "us",
        "is",
        "are",
        "was",
        "were",
        "been",
        "has",
        "had",
        "did",
        "does",
        "am",
    }
)


@dataclass
class DedupResult:
    original_count: int
    deduplicated_count: int
    duplicates_removed: int
    duplicate_ratio: float


class TextDeduplicator:
    """Deduplicate text datasets using MinHash LSH or exact hashing."""

    @staticmethod
    def exact_dedup(texts: list[str]) -> tuple[list[str], DedupResult]:
        if not texts:
            raise DataValidationError("Input texts list is empty")
        seen: set[str] = set()
        result: list[str] = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).hexdigest()
            if h not in seen:
                seen.add(h)
                result.append(t)
        n = len(texts)
        d = len(result)
        return result, DedupResult(n, d, n - d, (n - d) / n if n else 0.0)

    @staticmethod
    def ngram_dedup(
        texts: list[str],
        n: int = 5,
        threshold: float = 0.8,
        num_perm: int = 128,
    ) -> tuple[list[str], DedupResult]:
        if not texts:
            raise DataValidationError("Input texts list is empty")
        original_count = len(texts)

        ngram_sets = [TextDeduplicator._char_ngrams(t, n) for t in texts]

        try:
            kept = TextDeduplicator._dedup_datasketch(texts, ngram_sets, threshold, num_perm)
        except ImportError:
            kept = TextDeduplicator._dedup_fallback(texts, ngram_sets, threshold, num_perm)

        d = len(kept)
        return kept, DedupResult(
            original_count,
            d,
            original_count - d,
            (original_count - d) / original_count if original_count else 0.0,
        )

    @staticmethod
    def _dedup_datasketch(
        texts: list[str],
        ngram_sets: list[set[str]],
        threshold: float,
        num_perm: int,
    ) -> list[str]:
        from datasketch import MinHash, MinHashLSH

        lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
        minhashes: list[MinHash] = []
        for i, ngs in enumerate(ngram_sets):
            m = MinHash(num_perm=num_perm)
            for ng in ngs:
                m.update(ng.encode("utf-8"))
            minhashes.append(m)
            lsh.insert(str(i), m)

        removed: set[int] = set()
        for i in range(len(texts)):
            if i in removed:
                continue
            candidates = lsh.query(minhashes[i])
            for c in candidates:
                j = int(c)
                if j <= i or j in removed:
                    continue
                sim = TextDeduplicator._jaccard_similarity(ngram_sets[i], ngram_sets[j])
                if sim >= threshold:
                    victim = j if len(texts[j]) <= len(texts[i]) else i
                    removed.add(victim)
                    if victim == i:
                        break

        return [t for idx, t in enumerate(texts) if idx not in removed]

    @staticmethod
    def _dedup_fallback(
        texts: list[str],
        ngram_sets: list[set[str]],
        threshold: float,
        num_perm: int,
    ) -> list[str]:
        sigs = [TextDeduplicator._minhash_signature(ngs, num_perm) for ngs in ngram_sets]

        removed: set[int] = set()
        for i in range(len(texts)):
            if i in removed:
                continue
            for j in range(i + 1, len(texts)):
                if j in removed:
                    continue
                # Quick minhash pre-filter
                matches = sum(1 for a, b in zip(sigs[i], sigs[j]) if a == b)
                est_sim = matches / num_perm
                if est_sim < threshold * 0.8:
                    continue
                sim = TextDeduplicator._jaccard_similarity(ngram_sets[i], ngram_sets[j])
                if sim >= threshold:
                    victim = j if len(texts[j]) <= len(texts[i]) else i
                    removed.add(victim)
                    if victim == i:
                        break

        return [t for idx, t in enumerate(texts) if idx not in removed]

    @staticmethod
    def _char_ngrams(text: str, n: int) -> set[str]:
        if len(text) < n:
            return set()
        return {text[i : i + n] for i in range(len(text) - n + 1)}

    @staticmethod
    def _jaccard_similarity(set_a: set, set_b: set) -> float:
        if not set_a and not set_b:
            return 1.0
        if not set_a or not set_b:
            return 0.0
        return len(set_a & set_b) / len(set_a | set_b)

    @staticmethod
    def _minhash_signature(ngrams: set[str], num_perm: int, seed: int = 42) -> list[int]:
        if not ngrams:
            return [0] * num_perm
        sig: list[int] = []
        for p in range(num_perm):
            min_h = min(hash(str(seed + p) + ng) for ng in ngrams)
            sig.append(min_h)
        return sig


class TextFilter:
    """Filter low-quality texts from training data."""

    @staticmethod
    def filter_by_length(
        texts: list[str],
        min_chars: int = 50,
        max_chars: int = 100_000,
        min_words: int = 10,
    ) -> tuple[list[str], dict]:
        if not texts:
            raise DataValidationError("Input texts list is empty")
        kept: list[str] = []
        too_short = 0
        too_long = 0
        for t in texts:
            nc = len(t)
            nw = len(t.split())
            if nc < min_chars or nw < min_words:
                too_short += 1
            elif nc > max_chars:
                too_long += 1
            else:
                kept.append(t)
        return kept, {
            "removed": too_short + too_long,
            "too_short": too_short,
            "too_long": too_long,
        }

    @staticmethod
    def filter_by_language(
        texts: list[str],
        target_lang: str = "en",
        min_confidence: float = 0.8,
    ) -> tuple[list[str], dict]:
        if not texts:
            raise DataValidationError("Input texts list is empty")

        kept: list[str] = []
        detected_langs: Counter[str] = Counter()
        removed = 0

        for t in texts:
            words = t.lower().split()
            if not words:
                removed += 1
                detected_langs["unknown"] += 1
                continue

            if target_lang == "en":
                stop_hits = sum(1 for w in words if w in _ENGLISH_STOP_WORDS)
                ratio = stop_hits / len(words)
                is_target = ratio >= (1.0 - min_confidence)
                lang = "en" if is_target else "other"
            else:
                ascii_count = sum(1 for c in t if ord(c) < 128)
                ascii_ratio = ascii_count / len(t) if t else 0.0
                is_target = ascii_ratio < 0.9
                lang = target_lang if is_target else "en"

            detected_langs[lang] += 1
            if is_target:
                kept.append(t)
            else:
                removed += 1

        return kept, {"removed": removed, "detected_langs": dict(detected_langs)}

    @staticmethod
    def filter_by_quality(
        texts: list[str],
        min_unique_words_ratio: float = 0.1,
        max_repetition_rate: float = 0.5,
        max_special_char_ratio: float = 0.3,
    ) -> tuple[list[str], dict]:
        if not texts:
            raise DataValidationError("Input texts list is empty")

        kept: list[str] = []
        reasons: Counter[str] = Counter()

        for t in texts:
            words = t.lower().split()

            if words:
                unique_ratio = len(set(words)) / len(words)
            else:
                unique_ratio = 0.0
            if unique_ratio < min_unique_words_ratio:
                reasons["low_diversity"] += 1
                continue

            if len(words) >= 3:
                trigrams = [tuple(words[i : i + 3]) for i in range(len(words) - 2)]
                freq = Counter(trigrams)
                repeated = sum(c for c in freq.values() if c > 1)
                rep_rate = repeated / len(trigrams)
            else:
                rep_rate = 0.0
            if rep_rate > max_repetition_rate:
                reasons["high_repetition"] += 1
                continue

            if t:
                special = sum(1 for c in t if not c.isalnum() and not c.isspace())
                special_ratio = special / len(t)
            else:
                special_ratio = 0.0
            if special_ratio > max_special_char_ratio:
                reasons["high_special_chars"] += 1
                continue

            kept.append(t)

        return kept, {"removed": sum(reasons.values()), "reasons": dict(reasons)}

    @classmethod
    def full_pipeline(
        cls,
        texts: list[str],
        dedup: bool = True,
        dedup_threshold: float = 0.8,
        min_chars: int = 50,
        min_words: int = 10,
    ) -> tuple[list[str], dict]:
        if not texts:
            raise DataValidationError("Input texts list is empty")

        summary: dict = {"original_count": len(texts)}

        if dedup:
            texts, dedup_result = TextDeduplicator.exact_dedup(texts)
            summary["dedup"] = {
                "duplicates_removed": dedup_result.duplicates_removed,
                "duplicate_ratio": dedup_result.duplicate_ratio,
            }

        texts, length_stats = (
            cls.filter_by_length(
                texts,
                min_chars=min_chars,
                min_words=min_words,
            )
            if texts
            else (texts, {"removed": 0, "too_short": 0, "too_long": 0})
        )
        summary["length_filter"] = length_stats

        texts, quality_stats = (
            cls.filter_by_quality(texts)
            if texts
            else (
                texts,
                {"removed": 0, "reasons": {}},
            )
        )
        summary["quality_filter"] = quality_stats

        summary["final_count"] = len(texts)
        return texts, summary


if __name__ == "__main__":
    # Exact dedup
    texts = ["the cat sat", "hello world", "the cat sat", "foo bar", "hello world"]
    clean, result = TextDeduplicator.exact_dedup(texts)
    print(f"Exact dedup: {result.original_count} → {result.deduplicated_count}")
    print(f"  Kept: {clean}")

    # Near-duplicate dedup (these two are very similar)
    near = [
        "The quick brown fox jumps over the lazy dog",
        "The quick brown fox jumped over the lazy dog",  # 1 word different
        "Something completely different about cats and dogs",
    ]
    clean2, result2 = TextDeduplicator.ngram_dedup(near, threshold=0.7)
    print(f"\nNear dedup (0.7): {result2.original_count} → {result2.deduplicated_count}")

    # Quality filter
    mixed = ["A good sentence with plenty of unique interesting words", "the the the the the"]
    filtered, stats = TextFilter.filter_by_quality(mixed, min_unique_words_ratio=0.3)
    print(f"\nQuality filter: kept {len(filtered)}/{len(mixed)}, removed for: {stats['reasons']}")
