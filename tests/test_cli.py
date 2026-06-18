# -*- coding: utf-8 -*-
"""
Test suite for the scomp-link CLI
"""
import pytest
import numpy as np
import pandas as pd
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from scomp_link.cli import main, build_parser


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def regression_csv(tmp_dir):
    np.random.seed(42)
    df = pd.DataFrame({'x1': np.random.randn(100), 'x2': np.random.randn(100),
                       'y': 2 * np.random.randn(100) + 0.5})
    path = tmp_dir / "regression.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def classification_csv(tmp_dir):
    np.random.seed(42)
    df = pd.DataFrame({'x1': np.random.randn(100), 'x2': np.random.randn(100),
                       'y': np.random.choice([0, 1], 100)})
    path = tmp_dir / "classification.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def parquet_file(tmp_dir):
    np.random.seed(42)
    df = pd.DataFrame({'a': np.random.randn(50), 'b': np.random.randn(50),
                       'target': np.random.randn(50)})
    path = tmp_dir / "data.parquet"
    df.to_parquet(path, index=False)
    return str(path)


@pytest.fixture
def artifact_path(regression_csv, tmp_dir):
    """Create a .scomp artifact for testing predict/explain/info."""
    from scomp_link import ScompArtifact
    from sklearn.linear_model import LinearRegression
    df = pd.read_csv(regression_csv)
    X = df[['x1', 'x2']]
    y = df['y']
    model = LinearRegression().fit(X, y)
    artifact = ScompArtifact()
    artifact.set_model(model)
    artifact.set_config(task_type='regression', target_col='y')
    artifact.set_metrics({'r2': 0.5})
    artifact.set_feature_schema(X)
    artifact.set_sample_data(X)
    path = str(tmp_dir / "test_model.scomp")
    artifact.save(path)
    return path


def run_cli(*args):
    """Helper to run CLI with given args, returns exit code."""
    with patch('sys.argv', ['scomp-link'] + list(args)):
        try:
            main()
            return 0
        except SystemExit as e:
            return e.code or 0


# ===================== PARSER =====================

class TestParser:

    def test_no_args_shows_help(self, capsys):
        run_cli()
        captured = capsys.readouterr()
        assert "Available commands" in captured.out

    def test_version(self, capsys):
        with pytest.raises(SystemExit):
            with patch('sys.argv', ['scomp-link', '--version']):
                main()

    def test_unknown_command(self):
        code = run_cli("nonexistent")
        assert code != 0


# ===================== RUN COMMAND =====================

class TestRunCommand:

    def test_run_regression(self, regression_csv, tmp_dir):
        output = str(tmp_dir / "results.json")
        code = run_cli("run", "--data", regression_csv, "--target", "y",
                       "--task", "regression", "--output", output, "--silent")
        assert code == 0
        assert os.path.exists(output)
        with open(output) as f:
            results = json.load(f)
        assert results['status'] == 'success'
        assert 'metrics' in results

    def test_run_classification(self, classification_csv, tmp_dir):
        output = str(tmp_dir / "results.json")
        code = run_cli("run", "--data", classification_csv, "--target", "y",
                       "--task", "classification", "--output", output, "--silent")
        assert code == 0
        with open(output) as f:
            results = json.load(f)
        assert results['status'] == 'success'

    def test_run_with_artifact(self, regression_csv, tmp_dir):
        artifact = str(tmp_dir / "model.scomp")
        code = run_cli("run", "--data", regression_csv, "--target", "y",
                       "--task", "regression", "--save-artifact", artifact, "--silent")
        assert code == 0
        assert os.path.exists(artifact)

    def test_run_parquet_input(self, parquet_file, tmp_dir):
        output = str(tmp_dir / "results.json")
        code = run_cli("run", "--data", parquet_file, "--target", "target",
                       "--task", "regression", "--output", output, "--silent")
        assert code == 0

    def test_run_with_features(self, regression_csv, tmp_dir):
        output = str(tmp_dir / "results.json")
        code = run_cli("run", "--data", regression_csv, "--target", "y",
                       "--task", "regression", "--features", "x1",
                       "--output", output, "--silent")
        assert code == 0

    def test_run_missing_file(self):
        code = run_cli("run", "--data", "/nonexistent.csv", "--target", "y",
                       "--task", "regression", "--silent")
        assert code != 0

    def test_run_missing_target(self, regression_csv):
        code = run_cli("run", "--data", regression_csv, "--target", "nonexistent",
                       "--task", "regression", "--silent")
        assert code != 0

    def test_run_stdout_output(self, regression_csv, capsys):
        code = run_cli("run", "--data", regression_csv, "--target", "y",
                       "--task", "regression", "--silent")
        assert code == 0
        captured = capsys.readouterr()
        output = json.loads(captured.out)
        assert output['status'] == 'success'


# ===================== PREDICT COMMAND =====================

class TestPredictCommand:

    def test_predict_csv_output(self, artifact_path, regression_csv, tmp_dir):
        output = str(tmp_dir / "preds.csv")
        code = run_cli("predict", "--artifact", artifact_path, "--data", regression_csv,
                       "--output", output, "--silent")
        assert code == 0
        df = pd.read_csv(output)
        assert 'prediction' in df.columns
        assert len(df) == 100

    def test_predict_default_output(self, artifact_path, regression_csv, tmp_dir, monkeypatch):
        monkeypatch.chdir(tmp_dir)
        code = run_cli("predict", "--artifact", artifact_path, "--data", regression_csv, "--silent")
        assert code == 0
        assert os.path.exists(tmp_dir / "predictions.csv")

    def test_predict_invalid_artifact(self, regression_csv):
        code = run_cli("predict", "--artifact", "/nonexistent.scomp",
                       "--data", regression_csv, "--silent")
        assert code != 0


# ===================== EXPLAIN COMMAND =====================

class TestExplainCommand:

    def test_explain_output(self, artifact_path, regression_csv, tmp_dir):
        output = str(tmp_dir / "importance.csv")
        code = run_cli("explain", "--artifact", artifact_path, "--data", regression_csv,
                       "--n-samples", "20", "--output", output, "--silent")
        assert code == 0
        df = pd.read_csv(output)
        assert 'feature' in df.columns
        assert 'mean_abs_shap' in df.columns


# ===================== DRIFT COMMAND =====================

class TestDriftCommand:

    def test_drift_no_drift(self, regression_csv, tmp_dir):
        output = str(tmp_dir / "drift.csv")
        code = run_cli("drift", "--reference", regression_csv, "--current", regression_csv,
                       "--output", output, "--silent")
        assert code == 0
        df = pd.read_csv(output)
        assert df['drifted'].sum() == 0

    def test_drift_with_drift(self, regression_csv, tmp_dir):
        # Create drifted data
        drifted_path = str(tmp_dir / "drifted.csv")
        np.random.seed(99)
        pd.DataFrame({'x1': np.random.randn(100) + 5, 'x2': np.random.randn(100),
                      'y': np.random.randn(100)}).to_csv(drifted_path, index=False)

        output = str(tmp_dir / "drift.csv")
        code = run_cli("drift", "--reference", regression_csv, "--current", drifted_path,
                       "--output", output, "--silent")
        assert code == 0
        df = pd.read_csv(output)
        assert df[df['feature'] == 'x1']['drifted'].values[0] == True

    def test_drift_with_features_filter(self, regression_csv, tmp_dir):
        output = str(tmp_dir / "drift.csv")
        code = run_cli("drift", "--reference", regression_csv, "--current", regression_csv,
                       "--features", "x1,x2", "--output", output, "--silent")
        assert code == 0
        df = pd.read_csv(output)
        assert len(df) == 2

    def test_drift_stdout(self, regression_csv, capsys):
        code = run_cli("drift", "--reference", regression_csv, "--current", regression_csv, "--silent")
        assert code == 0
        captured = capsys.readouterr()
        assert "feature" in captured.out


# ===================== QUALITY COMMAND =====================

class TestQualityCommand:

    def test_quality_html(self, regression_csv, tmp_dir):
        output = str(tmp_dir / "report.html")
        code = run_cli("quality", "--data", regression_csv, "--output", output, "--silent")
        assert code == 0
        assert os.path.exists(output)
        assert os.path.getsize(output) > 100

    def test_quality_csv(self, regression_csv, tmp_dir):
        output = str(tmp_dir / "report.csv")
        code = run_cli("quality", "--data", regression_csv, "--output", output, "--silent")
        assert code == 0
        assert os.path.exists(output)


# ===================== INFO COMMAND =====================

class TestInfoCommand:

    def test_info(self, artifact_path, capsys):
        code = run_cli("info", "--artifact", artifact_path)
        assert code == 0
        captured = capsys.readouterr()
        info = json.loads(captured.out)
        assert info['model_type'] == 'LinearRegression'
        assert info['has_model'] == True
        assert info['config']['task_type'] == 'regression'

    def test_info_invalid_file(self, tmp_dir):
        fake = str(tmp_dir / "fake.scomp")
        with open(fake, 'w') as f:
            f.write("not a scomp file")
        code = run_cli("info", "--artifact", fake)
        assert code != 0
