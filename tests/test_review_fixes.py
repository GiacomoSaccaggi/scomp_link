# -*- coding: utf-8 -*-
"""
Regression tests for the correctness and security fixes found during the
end-to-end review.

Each test here maps to a defect that shipped in 2.2.1, so they are written to
fail loudly if the old behaviour ever returns.
"""

import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from scomp_link import ScompLinkPipeline
from scomp_link.exceptions import ArtifactError
from scomp_link.persistence.artifact import ScompArtifact
from scomp_link.preprocessing.data_processor import Preprocessor, build_feature_pipeline


@pytest.fixture
def mixed_df():
    """Tabular data with a string column, a bool column and NaNs."""
    rng = np.random.default_rng(42)
    n = 300
    df = pd.DataFrame(
        {
            "sqm": rng.normal(90, 25, n),
            "rooms": rng.integers(1, 6, n),
            "zone": rng.choice(["A", "B", "C"], n),
            "premium": rng.choice([True, False], n),
        }
    )
    df["price"] = df.sqm * 1200 + df.rooms * 5000 + rng.normal(0, 5000, n)
    df.loc[0:9, "sqm"] = np.nan
    return df


# ---------------------------------------------------------------------------
# build_feature_pipeline
# ---------------------------------------------------------------------------


class TestBuildFeaturePipeline:
    def test_encodes_categoricals_and_imputes_nans(self, mixed_df):
        X = mixed_df.drop(columns=["price"])
        pre = build_feature_pipeline(X)
        out = pre.fit_transform(X)
        assert not np.isnan(np.asarray(out, dtype=float)).any(), "NaNs must be imputed"
        # 2 numeric + 1 bool + zone one-hot. zone contributes 2 columns when the
        # first level is dropped (preferred) or 3 on scikit-learn versions that
        # reject drop with handle_unknown="ignore".
        assert out.shape[1] in (5, 6), f"unexpected encoded width: {out.shape[1]}"
        assert out.shape[1] == len(pre.get_feature_names_out())

    def test_unknown_category_at_transform_time_is_ignored(self, mixed_df):
        X = mixed_df.drop(columns=["price"])
        pre = build_feature_pipeline(X).fit(X)
        width = len(pre.get_feature_names_out())
        unseen = pd.DataFrame([{"sqm": 100.0, "rooms": 3, "zone": "NEVER_SEEN", "premium": True}])
        out = pre.transform(unseen)
        assert out.shape == (1, width)
        assert np.isfinite(np.asarray(out, dtype=float)).all()

    def test_categorical_encoding_avoids_collinearity_when_supported(self, mixed_df):
        """A full one-hot expansion makes linear-model coefficients degenerate."""
        X = mixed_df.drop(columns=["price"])
        names = list(build_feature_pipeline(X).fit(X).get_feature_names_out())
        zone_cols = [n for n in names if "zone" in n]
        assert len(zone_cols) in (2, 3)

    def test_numeric_only_frame_is_supported(self):
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        assert build_feature_pipeline(X).fit_transform(X).shape == (3, 2)

    def test_rejects_frame_without_usable_columns(self):
        X = pd.DataFrame({"when": pd.to_datetime(["2024-01-01", "2024-01-02"])})
        with pytest.raises(ValueError, match="No usable feature columns"):
            build_feature_pipeline(X)


# ---------------------------------------------------------------------------
# prepare_datasets must honour feature_cols (silent data-leakage regression)
# ---------------------------------------------------------------------------


class TestPrepareDatasetsFeatureCols:
    def test_feature_cols_whitelist_is_respected(self):
        df = pd.DataFrame({"good": [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10], "leaky": range(10), "y": range(10)})
        X_train, X_test, _, _ = Preprocessor(df).prepare_datasets("y", feature_cols=["good"])
        assert list(X_train.columns) == ["good"]
        assert list(X_test.columns) == ["good"]

    def test_target_is_never_a_feature_even_if_listed(self):
        df = pd.DataFrame({"a": range(10), "y": range(10)})
        X_train, _, _, _ = Preprocessor(df).prepare_datasets("y", feature_cols=["a", "y"])
        assert "y" not in X_train.columns

    def test_all_columns_used_when_feature_cols_omitted(self):
        df = pd.DataFrame({"a": range(10), "b": range(10), "y": range(10)})
        X_train, _, _, _ = Preprocessor(df).prepare_datasets("y")
        assert sorted(X_train.columns) == ["a", "b"]

    def test_unknown_target_raises(self):
        with pytest.raises(ValueError, match="Target column 'nope' not found"):
            Preprocessor(pd.DataFrame({"a": range(5)})).prepare_datasets("nope")

    def test_unknown_feature_column_raises(self):
        df = pd.DataFrame({"a": range(10), "y": range(10)})
        with pytest.raises(ValueError, match="Feature columns not found"):
            Preprocessor(df).prepare_datasets("y", feature_cols=["ghost"])

    def test_feature_cols_containing_only_target_raises(self):
        df = pd.DataFrame({"a": range(10), "y": range(10)})
        with pytest.raises(ValueError, match="at least one column other than the target"):
            Preprocessor(df).prepare_datasets("y", feature_cols=["y"])


# ---------------------------------------------------------------------------
# run_pipeline on mixed-type data (the original crash)
# ---------------------------------------------------------------------------


class TestRunPipelineMixedTypes:
    def test_regression_trains_with_categorical_column(self, mixed_df):
        pipe = ScompLinkPipeline("mixed regression")
        pipe.import_and_clean_data(mixed_df)
        pipe.select_variables(target_col="price")
        pipe.choose_model("numerical_prediction")
        results = pipe.run_pipeline(task_type="regression")
        assert results["status"] == "success"
        assert results["metrics"]["r2"] > 0.5

    def test_fitted_model_predicts_on_raw_dataframe(self, mixed_df):
        pipe = ScompLinkPipeline("raw predict")
        pipe.import_and_clean_data(mixed_df)
        pipe.select_variables(target_col="price")
        pipe.choose_model("numerical_prediction")
        pipe.run_pipeline(task_type="regression")

        raw = pd.DataFrame([{"sqm": 100.0, "rooms": 3, "zone": "A", "premium": True}])
        preds = pipe.model.predict(raw)
        assert preds.shape == (1,)
        assert np.isfinite(preds[0])

    def test_excluded_column_does_not_reach_the_model(self):
        """A perfectly predictive column that was excluded must not be used."""
        rng = np.random.default_rng(0)
        n = 200
        df = pd.DataFrame({"signal": rng.normal(size=n)})
        df["y"] = df.signal * 3
        df["leak"] = df.y  # exact copy of the target

        pipe = ScompLinkPipeline("leak guard")
        pipe.import_and_clean_data(df)
        pipe.select_variables(target_col="y", feature_cols=["signal"])
        pipe.choose_model("numerical_prediction")
        pipe.run_pipeline(task_type="regression")

        step = pipe.model.named_steps["preprocessor"]
        used = [c for _, _, cols in step.transformers for c in cols]
        assert used == ["signal"], f"leaked columns reached the model: {used}"

    def test_clustering_handles_categorical_features(self, mixed_df):
        pipe = ScompLinkPipeline("mixed clustering")
        pipe.import_and_clean_data(mixed_df)
        pipe.select_variables(target_col="price")
        pipe.choose_model("categorical_known", {"n_categories": 3})
        results = pipe.run_pipeline(task_type="clustering", n_clusters=3)
        assert results["status"] == "success"
        assert results["n_clusters"] == 3

    def test_already_wrapped_model_is_not_double_wrapped(self, mixed_df):
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import Pipeline as SkPipeline

        X = mixed_df.drop(columns=["price"])
        existing = SkPipeline([("preprocessor", build_feature_pipeline(X)), ("model", Ridge())])
        assert ScompLinkPipeline._wrap_with_preprocessing(existing, X) is existing

    def test_non_sklearn_estimator_is_returned_untouched(self, mixed_df):
        sentinel = object()
        X = mixed_df.drop(columns=["price"])
        assert ScompLinkPipeline._wrap_with_preprocessing(sentinel, X) is sentinel


# ---------------------------------------------------------------------------
# silhouette guard
# ---------------------------------------------------------------------------


def test_silhouette_is_none_when_only_one_cluster_found():
    """Silhouette is undefined for a single label; it must not raise."""
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"a": rng.normal(size=60), "b": rng.normal(size=60), "target": 0})
    pipe = ScompLinkPipeline("single cluster")
    pipe.import_and_clean_data(df)
    pipe.select_variables(target_col="target", feature_cols=["a", "b"])
    pipe.choose_model("categorical_known", {"n_categories": 1})
    results = pipe.run_pipeline(task_type="clustering", n_clusters=1)
    assert results["status"] == "success"
    assert results["n_clusters"] == 1
    assert results["metrics"]["silhouette_score"] is None


def test_meanshift_survives_degenerate_identical_rows():
    """estimate_bandwidth returns 0 for identical rows; MeanShift rejects that."""
    df = pd.DataFrame({"a": [1.0] * 40, "b": [2.0] * 40, "target": 0})
    pipe = ScompLinkPipeline("degenerate meanshift")
    pipe.import_and_clean_data(df)
    pipe.select_variables(target_col="target", feature_cols=["a", "b"])
    pipe.choose_model("categorical_unknown", metadata={"categories_known": False})
    results = pipe.run_pipeline(task_type="clustering")
    assert results["status"] == "success"
    assert results["n_clusters"] >= 1


# ---------------------------------------------------------------------------
# artifact error handling
# ---------------------------------------------------------------------------


class TestArtifactErrors:
    def test_non_zip_file_raises_artifact_error(self, tmp_path):
        bad = tmp_path / "bad.scomp"
        bad.write_text("definitely not a zip")
        with pytest.raises(ArtifactError, match="not a valid .scomp artifact"):
            ScompArtifact.load(bad)

    def test_zip_without_magic_raises_artifact_error(self, tmp_path):
        import zipfile

        bad = tmp_path / "nomagic.scomp"
        with zipfile.ZipFile(bad, "w") as zf:
            zf.writestr("hello.txt", "world")
        with pytest.raises(ArtifactError, match="missing magic entry"):
            ScompArtifact.load(bad)

    def test_missing_file_still_raises_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            ScompArtifact.load(tmp_path / "absent.scomp")

    def test_roundtrip_artifact_predicts_on_raw_data(self, tmp_path, mixed_df):
        pipe = ScompLinkPipeline("artifact roundtrip")
        pipe.import_and_clean_data(mixed_df)
        pipe.select_variables(target_col="price")
        pipe.choose_model("numerical_prediction")
        pipe.run_pipeline(task_type="regression")

        art = ScompArtifact()
        art.set_model(pipe.model)
        art.set_config(task_type="regression", target_col="price")
        path = art.save(tmp_path / "m.scomp")

        loaded = ScompArtifact.load(path)
        raw = pd.DataFrame([{"sqm": 95.0, "rooms": 2, "zone": "C", "premium": False}])
        assert np.isfinite(loaded.predict(raw)[0])


# ---------------------------------------------------------------------------
# logging goes to stderr, not stdout
# ---------------------------------------------------------------------------


def test_library_logs_go_to_stderr_not_stdout():
    """stdout must stay clean so command output can be piped and parsed."""
    code = (
        "import scomp_link, pandas as pd;"
        "scomp_link.set_verbosity('info');"
        "from scomp_link.preprocessing.data_processor import Preprocessor;"
        "Preprocessor(pd.DataFrame({'a':[1.0,2,3],'y':[1.0,2,3]})).clean_data()"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == "", f"logs leaked to stdout: {proc.stdout!r}"
    assert "Cleaning data" in proc.stderr


# ---------------------------------------------------------------------------
# CLI error surface
# ---------------------------------------------------------------------------


class TestCliErrorHandling:
    def _run(self, *args):
        return subprocess.run([sys.executable, "-m", "scomp_link.cli", *args], capture_output=True, text=True)

    def test_invalid_artifact_reports_clean_error_without_traceback(self, tmp_path):
        bad = tmp_path / "bad.scomp"
        bad.write_text("nope")
        data = tmp_path / "d.csv"
        pd.DataFrame({"a": [1, 2, 3]}).to_csv(data, index=False)

        proc = self._run("predict", "--artifact", str(bad), "--data", str(data))
        assert proc.returncode != 0
        combined = proc.stdout + proc.stderr
        assert "ArtifactError" in combined
        assert "Traceback (most recent call last)" not in combined

    def test_traceback_flag_restores_full_stack(self, tmp_path):
        bad = tmp_path / "bad.scomp"
        bad.write_text("nope")
        data = tmp_path / "d.csv"
        pd.DataFrame({"a": [1, 2, 3]}).to_csv(data, index=False)

        proc = self._run("predict", "--artifact", str(bad), "--data", str(data), "--traceback")
        assert proc.returncode != 0
        assert "Traceback (most recent call last)" in proc.stderr
