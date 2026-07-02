# -*- coding: utf-8 -*-
"""
Test suite for new CLI commands: engineer, forecast, anomaly, fairness, compare + run flags
"""
import pytest
import numpy as np
import pandas as pd
import os
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

from scomp_link.cli import main


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def regression_csv(tmp_dir):
    np.random.seed(42)
    df = pd.DataFrame({'x1': np.random.randn(100), 'x2': np.random.randn(100),
                       'y': 2 * np.random.randn(100) + 0.5})
    path = tmp_dir / "data.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def series_csv(tmp_dir):
    np.random.seed(42)
    df = pd.DataFrame({'value': 50 + np.cumsum(np.random.randn(100))})
    path = tmp_dir / "series.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def fairness_csv(tmp_dir):
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        'y_true': np.random.binomial(1, 0.5, n),
        'y_pred': np.random.binomial(1, 0.5, n),
        'gender': np.random.choice(['M', 'F'], n),
    })
    path = tmp_dir / "fairness.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def two_artifacts(regression_csv, tmp_dir):
    from scomp_link import ScompArtifact
    from sklearn.linear_model import LinearRegression, Ridge
    df = pd.read_csv(regression_csv)
    X, y = df[['x1', 'x2']], df['y']

    paths = []
    for i, model_cls in enumerate([LinearRegression, Ridge]):
        m = model_cls().fit(X, y)
        a = ScompArtifact().set_model(m).set_config(task_type='regression', target_col='y')
        a.set_metrics({'r2': float(m.score(X, y))})
        p = str(tmp_dir / f"model_v{i+1}.scomp")
        a.save(p)
        paths.append(p)
    return paths


def run_cli(*args):
    with patch('sys.argv', ['scomp-link'] + list(args)):
        try:
            main()
            return 0
        except SystemExit as e:
            return e.code or 0


# ===================== ENGINEER =====================

class TestEngineerCommand:

    def test_engineer_interactions(self, regression_csv, tmp_dir):
        output = str(tmp_dir / "eng.csv")
        code = run_cli("engineer", "--data", regression_csv, "--target", "y",
                       "--interactions", "--output", output, "--silent")
        assert code == 0
        df = pd.read_csv(output)
        assert 'x1_x_x2' in df.columns
        assert 'y' in df.columns

    def test_engineer_log_transform(self, tmp_dir):
        np.random.seed(42)
        df = pd.DataFrame({'income': np.random.exponential(50000, 100), 'y': np.random.randn(100)})
        path = str(tmp_dir / "skewed.csv")
        df.to_csv(path, index=False)
        output = str(tmp_dir / "eng.csv")
        code = run_cli("engineer", "--data", path, "--target", "y",
                       "--log-transform", "--output", output, "--silent")
        assert code == 0
        result = pd.read_csv(output)
        assert 'income_log' in result.columns

    def test_engineer_parquet_output(self, regression_csv, tmp_dir):
        output = str(tmp_dir / "eng.parquet")
        code = run_cli("engineer", "--data", regression_csv, "--target", "y",
                       "--interactions", "--output", output, "--silent")
        assert code == 0
        df = pd.read_parquet(output)
        assert len(df) == 100

    def test_engineer_no_target(self, regression_csv, tmp_dir):
        output = str(tmp_dir / "eng.csv")
        code = run_cli("engineer", "--data", regression_csv,
                       "--interactions", "--output", output, "--silent")
        assert code == 0


# ===================== FORECAST =====================

class TestForecastCommand:

    def test_forecast_arima(self, series_csv, tmp_dir):
        output = str(tmp_dir / "fc.csv")
        code = run_cli("forecast", "--data", series_csv, "--column", "value",
                       "--horizon", "10", "--method", "arima", "--output", output, "--silent")
        assert code == 0
        df = pd.read_csv(output)
        assert len(df) == 10
        assert 'forecast' in df.columns
        assert 'lower' in df.columns
        assert 'upper' in df.columns

    def test_forecast_with_cv(self, series_csv, tmp_dir, capsys):
        output = str(tmp_dir / "fc.csv")
        code = run_cli("forecast", "--data", series_csv, "--column", "value",
                       "--horizon", "10", "--cv-splits", "3", "--output", output, "--silent")
        assert code == 0
        captured = capsys.readouterr()
        assert "Walk-forward CV" in captured.out

    def test_forecast_missing_column(self, series_csv):
        code = run_cli("forecast", "--data", series_csv, "--column", "nonexistent",
                       "--horizon", "5", "--silent")
        assert code != 0

    def test_forecast_auto_method(self, series_csv, tmp_dir):
        output = str(tmp_dir / "fc.csv")
        code = run_cli("forecast", "--data", series_csv, "--column", "value",
                       "--horizon", "5", "--method", "auto", "--output", output, "--silent")
        assert code == 0


# ===================== ANOMALY =====================

class TestAnomalyCommand:

    def test_anomaly_iforest_lof(self, regression_csv, tmp_dir):
        output = str(tmp_dir / "anom.csv")
        code = run_cli("anomaly", "--data", regression_csv, "--methods", "iforest,lof",
                       "--output", output, "--silent")
        assert code == 0
        df = pd.read_csv(output)
        assert 'is_anomaly' in df.columns

    def test_anomaly_custom_features(self, regression_csv, tmp_dir):
        output = str(tmp_dir / "anom.csv")
        code = run_cli("anomaly", "--data", regression_csv, "--features", "x1,x2",
                       "--contamination", "0.1", "--output", output, "--silent")
        assert code == 0

    def test_anomaly_stdout_comparison(self, regression_csv, tmp_dir, capsys):
        output = str(tmp_dir / "anom2.csv")
        code = run_cli("anomaly", "--data", regression_csv, "--methods", "iforest,lof",
                       "--output", output, "--silent")
        assert code == 0
        captured = capsys.readouterr()
        assert "iforest" in captured.out


# ===================== FAIRNESS =====================

class TestFairnessCommand:

    def test_fairness_stdout(self, fairness_csv, capsys):
        code = run_cli("fairness", "--data", fairness_csv, "--target", "y_true",
                       "--predicted", "y_pred", "--sensitive", "gender", "--silent")
        assert code == 0
        captured = capsys.readouterr()
        assert "Demographic Parity" in captured.out

    def test_fairness_json_output(self, fairness_csv, tmp_dir):
        output = str(tmp_dir / "fair.json")
        code = run_cli("fairness", "--data", fairness_csv, "--target", "y_true",
                       "--predicted", "y_pred", "--sensitive", "gender",
                       "--output", output, "--silent")
        assert code == 0
        with open(output) as f:
            report = json.load(f)
        assert 'demographic_parity' in report

    def test_fairness_missing_column(self, fairness_csv):
        code = run_cli("fairness", "--data", fairness_csv, "--target", "y_true",
                       "--predicted", "nonexistent", "--sensitive", "gender", "--silent")
        assert code != 0


# ===================== COMPARE =====================

class TestCompareCommand:

    def test_compare_stdout(self, two_artifacts, capsys):
        code = run_cli("compare", "--artifacts", *two_artifacts)
        assert code == 0
        captured = capsys.readouterr()
        assert "LinearRegression" in captured.out
        assert "Ridge" in captured.out

    def test_compare_csv_output(self, two_artifacts, tmp_dir):
        output = str(tmp_dir / "cmp.csv")
        code = run_cli("compare", "--artifacts", *two_artifacts, "--output", output)
        assert code == 0
        df = pd.read_csv(output)
        assert len(df) == 2
        assert 'r2' in df.columns

    def test_compare_missing_artifact(self, tmp_dir):
        code = run_cli("compare", "--artifacts", "/nonexistent.scomp")
        assert code != 0


# ===================== RUN FLAGS =====================

class TestRunFlags:

    def test_run_with_engineer(self, regression_csv, tmp_dir):
        output = str(tmp_dir / "results.json")
        code = run_cli("run", "--data", regression_csv, "--target", "y",
                       "--task", "regression", "--engineer", "--output", output, "--silent")
        assert code == 0
        with open(output) as f:
            r = json.load(f)
        assert r['status'] == 'success'

    def test_run_with_advanced_cv(self, regression_csv, tmp_dir):
        output = str(tmp_dir / "results.json")
        code = run_cli("run", "--data", regression_csv, "--target", "y",
                       "--task", "regression", "--advanced-cv", "--output", output, "--silent")
        assert code == 0


# ===================== INIT =====================

class TestInitCommand:

    def test_init_creates_project(self, tmp_dir):
        project = str(tmp_dir / "test_project")
        code = run_cli("init", project)
        assert code == 0
        assert os.path.exists(os.path.join(project, "pipeline.py"))
        assert os.path.exists(os.path.join(project, "config.yaml"))
        assert os.path.exists(os.path.join(project, "README.md"))
        assert os.path.exists(os.path.join(project, ".gitignore"))
        assert os.path.isdir(os.path.join(project, "data"))
        assert os.path.isdir(os.path.join(project, "models"))
        assert os.path.isdir(os.path.join(project, "reports"))

    def test_init_existing_dir_fails(self, tmp_dir):
        project = str(tmp_dir / "existing")
        os.makedirs(project)
        code = run_cli("init", project)
        assert code != 0

    def test_init_force_overwrites(self, tmp_dir):
        project = str(tmp_dir / "existing2")
        os.makedirs(project)
        code = run_cli("init", project, "--force")
        assert code == 0
        assert os.path.exists(os.path.join(project, "pipeline.py"))

    def test_init_pipeline_contains_project_name(self, tmp_dir):
        project = str(tmp_dir / "my_awesome_project")
        run_cli("init", project)
        with open(os.path.join(project, "pipeline.py")) as f:
            content = f.read()
        assert "my_awesome_project" in content


# ===================== REPORT =====================

class TestReportCommand:

    def test_report_eda(self, regression_csv, tmp_dir):
        output = str(tmp_dir / "eda.html")
        code = run_cli("report", "--data", regression_csv, "--output", output, "--silent")
        assert code == 0
        assert os.path.exists(output)
        with open(output) as f:
            html = f.read()
        assert "Distribution" in html
        assert "Correlation" in html

    def test_report_eda_no_data_fails(self):
        code = run_cli("report", "--output", "/tmp/x.html", "--silent")
        assert code != 0

    def test_report_model(self, regression_csv, tmp_dir):
        # Create artifact first
        from scomp_link import ScompArtifact
        from sklearn.linear_model import LinearRegression
        df = pd.read_csv(regression_csv)
        model = LinearRegression().fit(df[['x1', 'x2']], df['y'])
        artifact_path = str(tmp_dir / "m.scomp")
        ScompArtifact().set_model(model).set_config(task_type='regression', target_col='y') \
            .set_metrics({'r2': 0.5}).set_feature_schema(df[['x1', 'x2']]).save(artifact_path)

        output = str(tmp_dir / "model_report.html")
        code = run_cli("report", "--artifact", artifact_path, "--data", regression_csv,
                       "--output", output, "--silent")
        assert code == 0
        assert os.path.exists(output)
        with open(output) as f:
            html = f.read()
        assert "Metrics" in html

    def test_report_model_without_data(self, tmp_dir):
        from scomp_link import ScompArtifact
        from sklearn.linear_model import LinearRegression
        import numpy as np
        model = LinearRegression().fit(np.array([[1],[2],[3]]), [1,2,3])
        artifact_path = str(tmp_dir / "m2.scomp")
        ScompArtifact().set_model(model).set_config(task_type='regression').set_metrics({'r2': 0.99}).save(artifact_path)

        output = str(tmp_dir / "report.html")
        code = run_cli("report", "--artifact", artifact_path, "--output", output, "--silent")
        assert code == 0
        assert os.path.exists(output)


# ===================== NEW CLI COMMANDS (serve, pipeline, describe, export, text, cluster, tune, validate, monitor) =====================

class TestNewCLICommands:
    """Tests for CLI commands added in the coverage boost session."""

    @pytest.fixture
    def tmp_dir(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            yield Path(d)

    @pytest.fixture
    def regression_csv(self, tmp_dir):
        np.random.seed(42)
        n = 150
        df = pd.DataFrame({'x1': np.random.randn(n), 'x2': np.random.randn(n),
                           'y': 2 * np.random.randn(n) + 0.5})
        df['y'] = 2 * df['x1'] - df['x2'] + np.random.randn(n) * 0.3
        path = tmp_dir / "reg.csv"
        df.to_csv(path, index=False)
        return str(path)

    @pytest.fixture
    def classification_csv(self, tmp_dir):
        np.random.seed(42)
        n = 100
        df = pd.DataFrame({'x1': np.random.randn(n), 'x2': np.random.randn(n),
                           'y': np.random.choice([0, 1], n)})
        path = tmp_dir / "cls.csv"
        df.to_csv(path, index=False)
        return str(path)

    @pytest.fixture
    def text_csv(self, tmp_dir):
        texts = ['payment failed charge'] * 15 + ['crash error bug'] * 15 + ['reset password login'] * 15
        labels = ['billing'] * 15 + ['tech'] * 15 + ['account'] * 15
        df = pd.DataFrame({'msg': texts, 'cat': labels})
        path = tmp_dir / "text.csv"
        df.to_csv(path, index=False)
        return str(path)

    @pytest.fixture
    def artifact_path(self, tmp_dir, regression_csv):
        """Create a .scomp artifact for testing."""
        from scomp_link.cli import build_parser
        parser = build_parser()
        art_path = str(tmp_dir / "model.scomp")
        args = parser.parse_args(['run', '--data', regression_csv, '--target', 'y',
                                  '--task', 'regression', '--save-artifact', art_path, '--silent'])
        args.func(args)
        return art_path

    def test_describe_table(self, regression_csv):
        from scomp_link.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(['describe', '--data', regression_csv, '--format', 'table'])
        args.func(args)  # should print table without error

    def test_describe_csv(self, regression_csv, tmp_dir):
        from scomp_link.cli import build_parser
        parser = build_parser()
        out = str(tmp_dir / "desc.csv")
        args = parser.parse_args(['describe', '--data', regression_csv, '--format', 'csv', '--output', out])
        args.func(args)
        assert Path(out).exists()

    def test_describe_json(self, regression_csv):
        from scomp_link.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(['describe', '--data', regression_csv, '--format', 'json'])
        args.func(args)

    def test_export_pickle(self, artifact_path, tmp_dir):
        from scomp_link.cli import build_parser
        parser = build_parser()
        out = str(tmp_dir / "model.pkl")
        args = parser.parse_args(['export', '--artifact', artifact_path, '--format', 'pickle', '--output', out])
        args.func(args)
        assert Path(out).exists()
        assert Path(out).stat().st_size > 0

    def test_export_joblib(self, artifact_path, tmp_dir):
        from scomp_link.cli import build_parser
        parser = build_parser()
        out = str(tmp_dir / "model.joblib")
        args = parser.parse_args(['export', '--artifact', artifact_path, '--format', 'joblib', '--output', out])
        args.func(args)
        assert Path(out).exists()

    def test_validate(self, artifact_path, regression_csv, tmp_dir):
        from scomp_link.cli import build_parser
        parser = build_parser()
        out = str(tmp_dir / "metrics.json")
        args = parser.parse_args(['validate', '--artifact', artifact_path, '--data', regression_csv,
                                  '--target', 'y', '--output', out, '--silent'])
        args.func(args)
        assert Path(out).exists()
        import json
        with open(out) as f:
            data = json.loads(f.read())
        assert 'metrics' in data
        assert 'r2' in data['metrics']

    def test_validate_with_report(self, artifact_path, regression_csv, tmp_dir):
        from scomp_link.cli import build_parser
        parser = build_parser()
        report = str(tmp_dir / "val_report.html")
        args = parser.parse_args(['validate', '--artifact', artifact_path, '--data', regression_csv,
                                  '--target', 'y', '--report', report, '--silent'])
        args.func(args)
        assert Path(report).exists()
        assert Path(report).stat().st_size > 1000

    def test_tune_optuna(self, regression_csv, tmp_dir):
        from scomp_link.cli import build_parser
        parser = build_parser()
        out = str(tmp_dir / "tune_results.json")
        args = parser.parse_args(['tune', '--data', regression_csv, '--target', 'y',
                                  '--task', 'regression', '--method', 'optuna', '--n-trials', '3',
                                  '--output', out, '--silent'])
        args.func(args)
        assert Path(out).exists()

    def test_tune_halving(self, regression_csv, tmp_dir):
        from scomp_link.cli import build_parser
        parser = build_parser()
        art = str(tmp_dir / "tuned.scomp")
        args = parser.parse_args(['tune', '--data', regression_csv, '--target', 'y',
                                  '--task', 'regression', '--method', 'halving',
                                  '--save-artifact', art, '--silent'])
        args.func(args)
        assert Path(art).exists()

    def test_tune_classification(self, classification_csv, tmp_dir):
        from scomp_link.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(['tune', '--data', classification_csv, '--target', 'y',
                                  '--task', 'classification', '--method', 'optuna', '--n-trials', '3',
                                  '--silent'])
        args.func(args)

    def test_cluster_kmeans(self, regression_csv, tmp_dir):
        from scomp_link.cli import build_parser
        parser = build_parser()
        out = str(tmp_dir / "clusters.csv")
        args = parser.parse_args(['cluster', '--data', regression_csv, '--n-clusters', '3',
                                  '--output', out, '--silent'])
        args.func(args)
        assert Path(out).exists()

    def test_cluster_with_plot(self, regression_csv, tmp_dir):
        from scomp_link.cli import build_parser
        parser = build_parser()
        plot = str(tmp_dir / "clusters.html")
        args = parser.parse_args(['cluster', '--data', regression_csv, '--n-clusters', '2',
                                  '--plot', plot, '--silent'])
        args.func(args)
        assert Path(plot).exists()

    def test_text_tfidf(self, text_csv):
        from scomp_link.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(['text', '--data', text_csv, '--text-col', 'msg',
                                  '--target', 'cat', '--method', 'tfidf', '--silent'])
        args.func(args)

    def test_monitor(self, tmp_dir):
        np.random.seed(42)
        ref = pd.DataFrame({'a': np.random.randn(100), 'b': np.random.randn(100)})
        cur = pd.DataFrame({'a': np.random.randn(50) + 2, 'b': np.random.randn(50)})
        ref_path = str(tmp_dir / "ref.csv")
        cur_path = str(tmp_dir / "cur.csv")
        ref.to_csv(ref_path, index=False)
        cur.to_csv(cur_path, index=False)

        from scomp_link.cli import build_parser
        parser = build_parser()
        out = str(tmp_dir / "monitor.html")
        args = parser.parse_args(['monitor', '--reference', ref_path, '--current', cur_path,
                                  '--output', out, '--silent'])
        args.func(args)
        assert Path(out).exists()
        assert Path(out).stat().st_size > 1000

    def test_pipeline_yaml(self, regression_csv, tmp_dir):
        import yaml
        config = {
            'data': regression_csv,
            'target': 'y',
            'task': 'regression',
            'tune': {'method': 'optuna', 'n_trials': 3},
            'save': str(tmp_dir / "pipe_model.scomp"),
        }
        cfg_path = str(tmp_dir / "pipeline.yaml")
        with open(cfg_path, 'w') as f:
            yaml.dump(config, f)

        from scomp_link.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(['pipeline', '--config', cfg_path, '--silent'])
        args.func(args)
        assert Path(config['save']).exists()

    def test_list_models(self):
        from scomp_link.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(['list-models'])
        args.func(args)  # should print without error

    def test_check_deps(self):
        from scomp_link.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(['check-deps'])
        args.func(args)  # should print without error

    def test_run_format_table(self, regression_csv):
        from scomp_link.cli import build_parser
        parser = build_parser()
        args = parser.parse_args(['run', '--data', regression_csv, '--target', 'y',
                                  '--task', 'regression', '--format', 'table', '--silent'])
        args.func(args)

    def test_run_format_csv(self, regression_csv, tmp_dir):
        from scomp_link.cli import build_parser
        parser = build_parser()
        out = str(tmp_dir / "results.csv")
        args = parser.parse_args(['run', '--data', regression_csv, '--target', 'y',
                                  '--task', 'regression', '--format', 'csv', '--output', out, '--silent'])
        args.func(args)
        assert Path(out).exists()
