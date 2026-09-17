# -*- coding: utf-8 -*-
"""
Coverage for modules that the suite previously did not reach:
``unsupervised_text`` (0%), ``anomaly_detector`` deep-learning branches,
and ``regressor_optimizer.Boruta``.

The sentence-transformer model is stubbed so these tests stay offline and fast;
the goal is to exercise scomp-link's own control flow, not to re-test upstream
embedding quality.
"""

import sys
import types

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# unsupervised_text.TextEmbeddingClustering  (was 0% covered)
# ---------------------------------------------------------------------------


@pytest.fixture
def stub_sentence_transformers(monkeypatch):
    """Install a deterministic fake SentenceTransformer in sys.modules."""

    class _FakeST:
        def __init__(self, model_name):
            self.model_name = model_name

        def encode(self, texts):
            # Deterministic 4-dim embedding keyed on text length and first char,
            # so semantically grouped inputs land in separable clusters.
            return np.array(
                [[len(t), ord(t[0]) % 7, len(t.split()), sum(map(ord, t)) % 11] for t in texts],
                dtype=float,
            )

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = _FakeST  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return module


TEXTS = [
    "cheap flight to rome",
    "cheap flight to milan",
    "budget flights italy",
    "refund my broken laptop",
    "laptop screen is broken",
    "broken device refund please",
]


class TestTextEmbeddingClustering:
    def test_fit_predict_kmeans_returns_one_label_per_document(self, stub_sentence_transformers):
        from scomp_link.models.unsupervised_text import TextEmbeddingClustering

        clus = TextEmbeddingClustering(n_clusters=2, method="kmeans")
        labels = clus.fit_predict(TEXTS)
        assert len(labels) == len(TEXTS)
        assert len(set(labels)) == 2

    def test_fit_then_predict_kmeans(self, stub_sentence_transformers):
        from scomp_link.models.unsupervised_text import TextEmbeddingClustering

        clus = TextEmbeddingClustering(n_clusters=2, method="kmeans").fit(TEXTS)
        labels = clus.predict(TEXTS)
        assert len(labels) == len(TEXTS)

    def test_hierarchical_method_selects_agglomerative(self, stub_sentence_transformers):
        from sklearn.cluster import AgglomerativeClustering

        from scomp_link.models.unsupervised_text import TextEmbeddingClustering

        clus = TextEmbeddingClustering(n_clusters=2, method="hierarchical")
        labels = clus.fit_predict(TEXTS)
        assert isinstance(clus.clusterer, AgglomerativeClustering)
        assert len(labels) == len(TEXTS)

    def test_predict_unsupported_for_agglomerative(self, stub_sentence_transformers):
        """AgglomerativeClustering has no predict(); the assert must catch it."""
        from scomp_link.models.unsupervised_text import TextEmbeddingClustering

        clus = TextEmbeddingClustering(n_clusters=2, method="hierarchical").fit(TEXTS)
        with pytest.raises(AssertionError):
            clus.predict(TEXTS)

    def test_model_is_lazily_initialised_once(self, stub_sentence_transformers):
        from scomp_link.models.unsupervised_text import TextEmbeddingClustering

        clus = TextEmbeddingClustering(n_clusters=2)
        assert clus.model is None
        clus.fit(TEXTS)
        first = clus.model
        clus.predict(TEXTS)
        assert clus.model is first, "model must be reused, not rebuilt"

    def test_default_params_are_exposed_for_sklearn_clone(self, stub_sentence_transformers):
        from sklearn.base import clone

        from scomp_link.models.unsupervised_text import TextEmbeddingClustering

        original = TextEmbeddingClustering(model_name="stub-model", n_clusters=3, method="kmeans")
        copy = clone(original)
        assert copy.model_name == "stub-model"
        assert copy.n_clusters == 3
        assert copy.method == "kmeans"


# ---------------------------------------------------------------------------
# anomaly_detector: deep-learning branches + reporting
# ---------------------------------------------------------------------------


@pytest.fixture
def anomaly_df():
    rng = np.random.default_rng(7)
    n = 120
    df = pd.DataFrame({"x": rng.normal(size=n), "y": rng.normal(size=n)})
    df.loc[:4, ["x", "y"]] = 9.0  # obvious outliers
    df["group"] = ["a"] * (n // 2) + ["b"] * (n - n // 2)
    return df


class TestAnomalyDetectorBranches:
    def test_transformer_autoencoder_flags_contaminated_fraction(self, anomaly_df):
        torch = pytest.importorskip("torch")
        assert torch is not None
        from scomp_link.models.anomaly_detector import AnomalyDetector

        det = AnomalyDetector(
            contamination=0.05,
            methods=["transformer"],
            transformer_epochs=2,
            transformer_d_model=8,
            transformer_nhead=2,
            transformer_num_layers=1,
            consensus_threshold=1,
            verbose=False,
        )
        res = det.fit_predict(anomaly_df, features=["x", "y"])
        assert "anom_transformer" in res["data"].columns
        assert res["data"]["anom_transformer"].sum() > 0

    def test_tabnet_autoencoder_runs_or_reports_missing_dependency(self, anomaly_df):
        from scomp_link.models.anomaly_detector import AnomalyDetector

        det = AnomalyDetector(
            contamination=0.05,
            methods=["tabnet"],
            tabnet_epochs=2,
            consensus_threshold=1,
            verbose=False,
        )
        try:
            res = det.fit_predict(anomaly_df, features=["x", "y"])
        except ImportError as exc:
            assert "pytorch-tabnet" in str(exc)
        else:
            assert "anom_tabnet" in res["data"].columns

    def test_report_groups_by_column(self, anomaly_df):
        from scomp_link.models.anomaly_detector import AnomalyDetector

        det = AnomalyDetector(contamination=0.05, methods=["iforest", "lof"], consensus_threshold=2, verbose=False)
        det.fit_predict(anomaly_df, features=["x", "y"])
        grouped = det.report(group_by=["group"])
        assert isinstance(grouped, pd.DataFrame)
        assert len(grouped) > 0

    def test_report_before_fit_raises(self):
        from scomp_link.models.anomaly_detector import AnomalyDetector

        det = AnomalyDetector(verbose=False)
        with pytest.raises((ValueError, AttributeError, AssertionError)):
            det.report()

    def test_rows_with_nan_features_are_dropped(self, anomaly_df):
        from scomp_link.models.anomaly_detector import AnomalyDetector

        df = anomaly_df.copy()
        df.loc[10:19, "x"] = np.nan
        det = AnomalyDetector(contamination=0.05, methods=["iforest"], consensus_threshold=1, verbose=False)
        res = det.fit_predict(df, features=["x", "y"])
        assert len(res["data"]) == len(df) - 10


# ---------------------------------------------------------------------------
# regressor_optimizer.Boruta
# ---------------------------------------------------------------------------


class TestBoruta:
    @pytest.fixture
    def xy(self):
        rng = np.random.default_rng(3)
        n = 120
        X = rng.normal(size=(n, 5))
        y = 4 * X[:, 0] - 3 * X[:, 1] + rng.normal(scale=0.2, size=n)
        return X, y

    def _estimator(self):
        from sklearn.ensemble import RandomForestRegressor

        return RandomForestRegressor(n_estimators=10, random_state=42)

    def test_fit_marks_informative_features_as_supported(self, xy):
        from scomp_link.models.regressor_optimizer import Boruta

        X, y = xy
        b = Boruta(self._estimator(), n_estimators=10, max_iter=8, random_state=42, verbose=0)
        b.fit(X, y)
        assert len(b.support_) == X.shape[1]
        assert len(b.ranking_) == X.shape[1]
        assert b.support_[0] or b.support_weak_[0], "the strongest feature should be retained"

    def test_transform_before_fit_raises_value_error(self, xy):
        from scomp_link.models.regressor_optimizer import Boruta

        X, _ = xy
        b = Boruta(self._estimator(), n_estimators=10, max_iter=5, random_state=42, verbose=0)
        with pytest.raises(ValueError, match="call the fit"):
            b.transform(X)

    def test_transform_reduces_feature_count(self, xy):
        from scomp_link.models.regressor_optimizer import Boruta

        X, y = xy
        b = Boruta(self._estimator(), n_estimators=10, max_iter=8, random_state=42, verbose=0)
        b.fit(X, y)
        reduced = b.transform(X)
        assert reduced.shape[0] == X.shape[0]
        assert reduced.shape[1] == int(np.sum(b.support_))

    def test_transform_weak_includes_tentative_features(self, xy):
        from scomp_link.models.regressor_optimizer import Boruta

        X, y = xy
        b = Boruta(self._estimator(), n_estimators=10, max_iter=8, random_state=42, verbose=0)
        b.fit(X, y)
        strong = b.transform(X).shape[1]
        weak = b.transform(X, weak=True).shape[1]
        assert weak >= strong

    def test_fit_transform_matches_separate_calls(self, xy):
        from scomp_link.models.regressor_optimizer import Boruta

        X, y = xy
        a = Boruta(self._estimator(), n_estimators=10, max_iter=8, random_state=42, verbose=0)
        b = Boruta(self._estimator(), n_estimators=10, max_iter=8, random_state=42, verbose=0)
        one_shot = a.fit_transform(X, y)
        b.fit(X, y)
        assert one_shot.shape == b.transform(X).shape
