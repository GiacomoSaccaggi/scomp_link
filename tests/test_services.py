# -*- coding: utf-8 -*-
"""Unit tests for scomp_link.services — all tests use synthetic data, no external files."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from scomp_link.exceptions import DataLoadError, DataValidationError, DriftDetectionError
from scomp_link.schemas import ClusterConfig, DescribeConfig, DriftConfig, EngineerConfig, TrainConfig
from scomp_link.services import cluster, describe, detect_drift, engineer, load_dataframe

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def regression_csv(tmp_path):
    np.random.seed(42)
    df = pd.DataFrame(
        {
            "x1": np.random.randn(100),
            "x2": np.random.randn(100),
            "x3": np.random.randn(100),
            "y": np.random.randn(100),
        }
    )
    p = tmp_path / "regression.csv"
    df.to_csv(p, index=False)
    return str(p)


@pytest.fixture
def regression_parquet(tmp_path):
    np.random.seed(42)
    df = pd.DataFrame({"a": np.random.randn(50), "b": np.random.randn(50), "y": np.random.randn(50)})
    p = tmp_path / "data.parquet"
    df.to_parquet(p, index=False)
    return str(p)


@pytest.fixture
def drift_ref_csv(tmp_path):
    np.random.seed(0)
    df = pd.DataFrame({"f1": np.random.randn(200), "f2": np.random.randn(200)})
    p = tmp_path / "ref.csv"
    df.to_csv(p, index=False)
    return str(p)


@pytest.fixture
def drift_cur_shifted_csv(tmp_path):
    """Distribution shifted by +5 — should trigger drift."""
    np.random.seed(1)
    df = pd.DataFrame({"f1": np.random.randn(200) + 5, "f2": np.random.randn(200) + 5})
    p = tmp_path / "cur.csv"
    df.to_csv(p, index=False)
    return str(p)


@pytest.fixture
def drift_cur_same_csv(tmp_path):
    """Same distribution as ref — should not trigger drift."""
    np.random.seed(99)
    df = pd.DataFrame({"f1": np.random.randn(200), "f2": np.random.randn(200)})
    p = tmp_path / "cur_same.csv"
    df.to_csv(p, index=False)
    return str(p)


@pytest.fixture
def cluster_csv(tmp_path):
    np.random.seed(42)
    # 3 clear clusters
    X = np.vstack(
        [
            np.random.randn(50, 2) + [0, 0],
            np.random.randn(50, 2) + [10, 10],
            np.random.randn(50, 2) + [0, 10],
        ]
    )
    df = pd.DataFrame(X, columns=["x", "y"])
    p = tmp_path / "clusters.csv"
    df.to_csv(p, index=False)
    return str(p)


# ── load_dataframe ─────────────────────────────────────────────────────────────


class TestLoadDataframe:
    def test_loads_csv(self, regression_csv):
        df = load_dataframe(regression_csv)
        assert len(df) == 100
        assert "x1" in df.columns

    def test_loads_parquet(self, regression_parquet):
        df = load_dataframe(regression_parquet)
        assert len(df) == 50
        assert "a" in df.columns

    def test_missing_file_raises_data_load_error(self, tmp_path):
        with pytest.raises(DataLoadError, match="not found"):
            load_dataframe(str(tmp_path / "nonexistent.csv"))

    def test_unsupported_format_raises_data_load_error(self, tmp_path):
        p = tmp_path / "data.xlsx"
        p.write_text("fake")
        with pytest.raises(DataLoadError, match="Unsupported file format"):
            load_dataframe(str(p))

    def test_returns_dataframe(self, regression_csv):
        result = load_dataframe(regression_csv)
        assert isinstance(result, pd.DataFrame)


# ── describe ──────────────────────────────────────────────────────────────────


class TestDescribe:
    def test_returns_shape(self, regression_csv):
        result = describe(DescribeConfig(data=regression_csv))
        assert result["shape"] == [100, 4]

    def test_returns_columns_list(self, regression_csv):
        result = describe(DescribeConfig(data=regression_csv))
        assert len(result["columns"]) == 4

    def test_numeric_cols_have_stats(self, regression_csv):
        result = describe(DescribeConfig(data=regression_csv))
        col = next(c for c in result["columns"] if c["column"] == "x1")
        assert "min" in col
        assert "max" in col
        assert "mean" in col
        assert "std" in col

    def test_missing_pct_is_zero_for_clean_data(self, regression_csv):
        result = describe(DescribeConfig(data=regression_csv))
        for col in result["columns"]:
            assert col["missing_pct"] == 0.0

    def test_bad_path_raises_data_load_error(self, tmp_path):
        with pytest.raises(DataLoadError):
            describe(DescribeConfig(data=str(tmp_path / "nope.csv")))


# ── detect_drift ──────────────────────────────────────────────────────────────


class TestDetectDrift:
    def test_no_drift_when_same_distribution(self, drift_ref_csv, drift_cur_same_csv):
        result = detect_drift(DriftConfig(reference=drift_ref_csv, current=drift_cur_same_csv))
        assert result["drifted_features"] == 0

    def test_drift_detected_when_shifted(self, drift_ref_csv, drift_cur_shifted_csv):
        result = detect_drift(DriftConfig(reference=drift_ref_csv, current=drift_cur_shifted_csv))
        assert result["drifted_features"] > 0

    def test_returns_expected_keys(self, drift_ref_csv, drift_cur_same_csv):
        result = detect_drift(DriftConfig(reference=drift_ref_csv, current=drift_cur_same_csv))
        assert "drifted_features" in result
        assert "total_features" in result
        assert "max_psi" in result

    def test_no_common_cols_raises_drift_error(self, tmp_path):
        ref = tmp_path / "ref.csv"
        cur = tmp_path / "cur.csv"
        pd.DataFrame({"a": [1, 2, 3]}).to_csv(ref, index=False)
        pd.DataFrame({"b": [1, 2, 3]}).to_csv(cur, index=False)
        with pytest.raises(DriftDetectionError):
            detect_drift(DriftConfig(reference=str(ref), current=str(cur)))

    def test_threshold_respected(self, drift_ref_csv, drift_cur_same_csv):
        # With very high threshold, the same-distribution data should not be flagged
        result = detect_drift(DriftConfig(reference=drift_ref_csv, current=drift_cur_same_csv, threshold=9.9))
        assert result["drifted_features"] == 0


# ── cluster ───────────────────────────────────────────────────────────────────


class TestCluster:
    def test_returns_expected_n_clusters(self, cluster_csv):
        result = cluster(ClusterConfig(data=cluster_csv, n_clusters=3))
        assert result["n_clusters"] == 3

    def test_silhouette_score_is_positive_for_clear_clusters(self, cluster_csv):
        result = cluster(ClusterConfig(data=cluster_csv, n_clusters=3))
        assert result["silhouette_score"] > 0

    def test_cluster_sizes_sum_to_total_rows(self, cluster_csv):
        result = cluster(ClusterConfig(data=cluster_csv, n_clusters=3))
        total = sum(result["cluster_sizes"].values())
        assert total == 150  # 3 * 50 rows

    def test_output_file_created(self, cluster_csv, tmp_path):
        out = str(tmp_path / "clustered.csv")
        cluster(ClusterConfig(data=cluster_csv, n_clusters=3, output=out))
        assert os.path.exists(out)
        df = pd.read_csv(out)
        assert "cluster" in df.columns

    def test_no_numeric_cols_raises_data_validation(self, tmp_path):
        p = tmp_path / "text_only.csv"
        pd.DataFrame({"name": ["Alice", "Bob", "Carol"]}).to_csv(p, index=False)
        with pytest.raises(DataValidationError):
            cluster(ClusterConfig(data=str(p), n_clusters=2))


# ── engineer ──────────────────────────────────────────────────────────────────


class TestEngineer:
    def test_returns_output_path(self, regression_csv, tmp_path):
        out = str(tmp_path / "eng.csv")
        result = engineer(EngineerConfig(data=regression_csv, target="y", output=out))
        assert result["output_path"] == out
        assert os.path.exists(out)

    def test_engineered_has_more_columns(self, regression_csv, tmp_path):
        out = str(tmp_path / "eng.csv")
        result = engineer(EngineerConfig(data=regression_csv, target="y", interactions=True, output=out))
        assert result["engineered_shape"][1] > result["original_shape"][1]

    def test_target_column_missing_raises_validation_error(self, regression_csv, tmp_path):
        out = str(tmp_path / "eng.csv")
        with pytest.raises(DataValidationError, match="not found"):
            engineer(EngineerConfig(data=regression_csv, target="nonexistent_col", output=out))
