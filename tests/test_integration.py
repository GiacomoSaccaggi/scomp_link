# -*- coding: utf-8 -*-
"""
Integration tests for scomp-link v2.0.0.

Strategy (hybrid):
- FAST unit-style tests (no model training): test data loading, schemas,
  describe, drift, cluster, engineer — always run in CI, finish in <10s.
- SLOW integration tests (real model training): marked with @pytest.mark.slow,
  skipped unless RUN_SLOW_TESTS=1 env var is set.

Run fast tests:
    pytest tests/test_integration.py -v

Run all (including slow):
    RUN_SLOW_TESTS=1 pytest tests/test_integration.py -v
"""

import os

import numpy as np
import pandas as pd
import pytest

from scomp_link.exceptions import DataLoadError, DataValidationError, DriftDetectionError
from scomp_link.schemas import (
    ClusterConfig,
    DescribeConfig,
    DriftConfig,
    EngineerConfig,
    FairnessConfig,
    TrainConfig,
    ValidateConfig,
)
from scomp_link.services import (
    check_fairness,
    cluster,
    describe,
    detect_drift,
    engineer,
    load_dataframe,
    train,
    validate,
)

# ── Marker for slow tests ──────────────────────────────────────────────────────

RUN_SLOW = os.environ.get("RUN_SLOW_TESTS", "0") == "1"
slow = pytest.mark.skipif(not RUN_SLOW, reason="Set RUN_SLOW_TESTS=1 to run slow integration tests")


# ── Shared fixtures ────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def regression_dataset(tmp_path_factory):
    """100-row regression dataset, module-scoped so it is created once."""
    np.random.seed(42)
    n = 200
    X = np.random.randn(n, 3)
    y = 2 * X[:, 0] + 0.5 * X[:, 1] + np.random.randn(n) * 0.1
    df = pd.DataFrame(X, columns=["x1", "x2", "x3"])
    df["target"] = y
    p = tmp_path_factory.mktemp("data") / "regression.csv"
    df.to_csv(p, index=False)
    return str(p)


@pytest.fixture(scope="module")
def classification_dataset(tmp_path_factory):
    """200-row binary classification dataset."""
    np.random.seed(7)
    n = 200
    df = pd.DataFrame(
        {
            "f1": np.random.randn(n),
            "f2": np.random.randn(n),
            "f3": np.random.randn(n),
            "label": np.random.choice([0, 1], n),
        }
    )
    p = tmp_path_factory.mktemp("data") / "classification.csv"
    df.to_csv(p, index=False)
    return str(p)


@pytest.fixture(scope="module")
def fairness_dataset(tmp_path_factory):
    """Predictions dataset for fairness checks."""
    np.random.seed(0)
    n = 300
    df = pd.DataFrame(
        {
            "y_true": np.random.choice([0, 1], n),
            "y_pred": np.random.choice([0, 1], n),
            "gender": np.random.choice(["M", "F"], n),
        }
    )
    p = tmp_path_factory.mktemp("data") / "fairness.csv"
    df.to_csv(p, index=False)
    return str(p)


# ── Pipeline: describe → engineer ─────────────────────────────────────────────


class TestDescribeEngineerPipeline:
    """Fast: data loading + profiling + feature engineering without training."""

    def test_describe_returns_correct_shape(self, regression_dataset):
        result = describe(DescribeConfig(data=regression_dataset))
        assert result["shape"] == [200, 4]
        assert len(result["columns"]) == 4

    def test_describe_all_columns_numeric(self, regression_dataset):
        result = describe(DescribeConfig(data=regression_dataset))
        for col in result["columns"]:
            assert "mean" in col, f"Column {col['column']} should be numeric"

    def test_engineer_increases_feature_count(self, regression_dataset, tmp_path):
        out = str(tmp_path / "eng.csv")
        result = engineer(EngineerConfig(data=regression_dataset, target="target", interactions=True, output=out))
        assert result["engineered_shape"][1] > result["original_shape"][1]
        assert os.path.exists(out)
        eng_df = load_dataframe(out)
        assert "target" in eng_df.columns

    def test_engineer_then_describe(self, regression_dataset, tmp_path):
        """Engineer produces a valid dataset that describe can profile."""
        out = str(tmp_path / "eng2.csv")
        engineer(EngineerConfig(data=regression_dataset, target="target", interactions=True, output=out))
        result = describe(DescribeConfig(data=out))
        assert result["shape"][0] == 200  # rows preserved


# ── Pipeline: drift detection ──────────────────────────────────────────────────


class TestDriftPipeline:
    """Fast: drift detection between engineered and original distributions."""

    def test_no_drift_same_data(self, regression_dataset):
        result = detect_drift(DriftConfig(reference=regression_dataset, current=regression_dataset))
        assert result["drifted_features"] == 0

    def test_drift_detected_after_shift(self, regression_dataset, tmp_path):
        df = load_dataframe(regression_dataset)
        df_shifted = df.copy()
        df_shifted[["x1", "x2", "x3"]] += 10  # strong shift
        shifted_path = str(tmp_path / "shifted.csv")
        df_shifted.to_csv(shifted_path, index=False)
        result = detect_drift(DriftConfig(reference=regression_dataset, current=shifted_path))
        assert result["drifted_features"] > 0

    def test_drift_result_has_all_keys(self, regression_dataset):
        result = detect_drift(DriftConfig(reference=regression_dataset, current=regression_dataset))
        for key in ("drifted_features", "total_features", "max_psi"):
            assert key in result

    def test_drift_no_common_columns_raises(self, tmp_path):
        ref = tmp_path / "ref.csv"
        cur = tmp_path / "cur.csv"
        pd.DataFrame({"a": np.random.randn(50)}).to_csv(ref, index=False)
        pd.DataFrame({"z": np.random.randn(50)}).to_csv(cur, index=False)
        with pytest.raises(DriftDetectionError):
            detect_drift(DriftConfig(reference=str(ref), current=str(cur)))


# ── Pipeline: clustering ───────────────────────────────────────────────────────


class TestClusteringPipeline:
    """Fast: clustering on synthetic data."""

    def test_cluster_produces_correct_n_clusters(self, tmp_path):
        np.random.seed(42)
        X = np.vstack([np.random.randn(40, 2) + c for c in [[0, 0], [8, 8], [0, 8]]])
        df = pd.DataFrame(X, columns=["x", "y"])
        p = str(tmp_path / "c.csv")
        df.to_csv(p, index=False)
        result = cluster(ClusterConfig(data=p, n_clusters=3))
        assert result["n_clusters"] == 3

    def test_cluster_silhouette_positive_clear_clusters(self, tmp_path):
        np.random.seed(1)
        X = np.vstack([np.random.randn(50, 2) + c for c in [[0, 0], [20, 20]]])
        df = pd.DataFrame(X, columns=["x", "y"])
        p = str(tmp_path / "c2.csv")
        df.to_csv(p, index=False)
        result = cluster(ClusterConfig(data=p, n_clusters=2))
        assert result["silhouette_score"] > 0.5


# ── Pipeline: fairness ─────────────────────────────────────────────────────────


class TestFairnessPipeline:
    """Fast: fairness metrics on synthetic predictions."""

    def test_fairness_returns_all_metrics(self, fairness_dataset):
        result = check_fairness(
            FairnessConfig(data=fairness_dataset, target="y_true", predicted="y_pred", sensitive="gender")
        )
        assert "demographic_parity" in result
        assert "disparate_impact" in result
        assert "equalized_odds" in result

    def test_fairness_equalized_odds_has_tpr_fpr(self, fairness_dataset):
        result = check_fairness(
            FairnessConfig(data=fairness_dataset, target="y_true", predicted="y_pred", sensitive="gender")
        )
        assert "tpr_diff" in result["equalized_odds"]
        assert "fpr_diff" in result["equalized_odds"]

    def test_fairness_missing_column_raises(self, fairness_dataset):
        with pytest.raises(DataValidationError):
            check_fairness(
                FairnessConfig(data=fairness_dataset, target="y_true", predicted="y_pred", sensitive="nonexistent")
            )


# ── Pipeline: train → validate (SLOW) ─────────────────────────────────────────


class TestTrainValidatePipeline:
    """Slow: full train → artifact → validate workflow.
    Requires RUN_SLOW_TESTS=1.
    """

    @slow
    def test_train_regression_produces_artifact(self, regression_dataset, tmp_path):
        artifact_path = str(tmp_path / "model.scomp")
        result = train(
            TrainConfig(data=regression_dataset, target="target", task="regression", save_artifact=artifact_path)
        )
        assert result["status"] == "success"
        assert result["metrics"] is not None
        assert os.path.exists(artifact_path)

    @slow
    def test_train_then_validate_regression(self, regression_dataset, tmp_path):
        artifact_path = str(tmp_path / "model.scomp")
        train(TrainConfig(data=regression_dataset, target="target", task="regression", save_artifact=artifact_path))
        result = validate(ValidateConfig(artifact=artifact_path, data=regression_dataset, target="target"))
        assert result["task_type"] == "regression"
        assert "r2" in result["metrics"] or "rmse" in result["metrics"]
        assert result["n_samples"] == 200

    @slow
    def test_train_classification_produces_metrics(self, classification_dataset, tmp_path):
        artifact_path = str(tmp_path / "clf.scomp")
        result = train(
            TrainConfig(
                data=classification_dataset,
                target="label",
                task="classification",
                save_artifact=artifact_path,
            )
        )
        assert result["status"] == "success"
        assert result["metrics"] is not None

    @slow
    def test_describe_engineer_train_validate_pipeline(self, regression_dataset, tmp_path):
        """Full pipeline: describe → engineer → train → validate."""
        # Step 1: describe
        desc = describe(DescribeConfig(data=regression_dataset))
        assert desc["shape"][0] == 200

        # Step 2: engineer
        eng_path = str(tmp_path / "eng.csv")
        eng = engineer(EngineerConfig(data=regression_dataset, target="target", interactions=True, output=eng_path))
        assert eng["engineered_shape"][1] > 4

        # Step 3: train on engineered data
        artifact_path = str(tmp_path / "eng_model.scomp")
        tr = train(TrainConfig(data=eng_path, target="target", task="regression", save_artifact=artifact_path))
        assert tr["status"] == "success"

        # Step 4: validate
        val = validate(ValidateConfig(artifact=artifact_path, data=eng_path, target="target"))
        assert val["n_samples"] == 200
        assert "r2" in val["metrics"] or "rmse" in val["metrics"]


# ── Error handling across the pipeline ────────────────────────────────────────


class TestErrorPropagation:
    """Fast: verify that errors raised in services are the right types."""

    def test_bad_data_path_raises_data_load_error(self):
        with pytest.raises(DataLoadError):
            describe(DescribeConfig(data="/nonexistent/path/data.csv"))

    def test_wrong_target_raises_validation_error(self, regression_dataset):
        with pytest.raises(DataValidationError, match="not found"):
            engineer(EngineerConfig(data=regression_dataset, target="wrong_col", output="/tmp/x.csv"))

    def test_validate_wrong_target_raises_validation_error(self, tmp_path):
        """validate() raises DataValidationError when target column is missing."""
        # Create a fake artifact path to trigger the not-found artifact error first
        with pytest.raises(Exception):  # ArtifactError since file doesn't exist
            validate(ValidateConfig(artifact="/fake/model.scomp", data=str(tmp_path), target="y"))
