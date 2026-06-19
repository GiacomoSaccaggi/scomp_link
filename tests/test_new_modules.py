# -*- coding: utf-8 -*-
"""
Test suite for new modules: Explainability, Advanced Tuning, Drift Detection, Persistence
"""
import pytest
import numpy as np
import pandas as pd
import os
import tempfile
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


@pytest.fixture
def regression_model_and_data():
    np.random.seed(42)
    X = pd.DataFrame({'a': np.random.randn(200), 'b': np.random.randn(200), 'c': np.random.randn(200)})
    y = 2 * X['a'] + 0.5 * X['b'] + np.random.randn(200) * 0.1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestRegressor(n_estimators=30, random_state=42)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


@pytest.fixture
def classification_model_and_data():
    np.random.seed(42)
    X = pd.DataFrame({'a': np.random.randn(200), 'b': np.random.randn(200), 'c': np.random.randn(200)})
    y = (X['a'] + X['b'] > 0).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(n_estimators=30, random_state=42)
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


# ===================== EXPLAINABILITY =====================

class TestShapExplainer:

    @pytest.fixture(autouse=True)
    def _skip_if_shap_broken(self):
        try:
            from scomp_link import ShapExplainer
            import shap
        except (ImportError, ValueError):
            pytest.skip("shap not available or incompatible (keras/transformers conflict)")

    def test_explain_returns_array(self, regression_model_and_data):
        from scomp_link import ShapExplainer
        model, X_train, X_test, _, _ = regression_model_and_data
        explainer = ShapExplainer(model, X_train[:50])
        values = explainer.explain(X_test[:10])
        assert values.shape == (10, 3)

    def test_feature_importance_sorted(self, regression_model_and_data):
        from scomp_link import ShapExplainer
        model, X_train, X_test, _, _ = regression_model_and_data
        explainer = ShapExplainer(model, X_train[:50])
        explainer.explain(X_test[:10])
        importance = explainer.feature_importance()
        assert importance.iloc[0]['feature'] == 'a'  # strongest predictor
        assert importance['mean_abs_shap'].is_monotonic_decreasing

    def test_feature_importance_before_explain_raises(self, regression_model_and_data):
        from scomp_link import ShapExplainer
        model, X_train, _, _, _ = regression_model_and_data
        explainer = ShapExplainer(model, X_train[:50])
        with pytest.raises(ValueError):
            explainer.feature_importance()

    def test_plot_importance_returns_figure(self, regression_model_and_data):
        from scomp_link import ShapExplainer
        model, X_train, X_test, _, _ = regression_model_and_data
        explainer = ShapExplainer(model, X_train[:50])
        explainer.explain(X_test[:10])
        fig = explainer.plot_importance(top_n=3)
        assert hasattr(fig, 'to_json')  # plotly figure


class TestLimeExplainer:

    def test_explain_instance_regression(self, regression_model_and_data):
        from scomp_link import LimeExplainer
        model, X_train, X_test, _, _ = regression_model_and_data
        explainer = LimeExplainer(model, X_train, task='regression')
        exp = explainer.explain_instance(X_test.iloc[0], num_features=3)
        assert len(exp.as_list()) == 3

    def test_explain_instance_classification(self, classification_model_and_data):
        from scomp_link import LimeExplainer
        model, X_train, X_test, _, _ = classification_model_and_data
        explainer = LimeExplainer(model, X_train, task='classification')
        exp = explainer.explain_instance(X_test.iloc[0], num_features=3)
        assert len(exp.as_list()) > 0

    def test_plot_explanation_returns_figure(self, regression_model_and_data):
        from scomp_link import LimeExplainer
        model, X_train, X_test, _, _ = regression_model_and_data
        explainer = LimeExplainer(model, X_train, task='regression')
        exp = explainer.explain_instance(X_test.iloc[0])
        fig = explainer.plot_explanation(exp)
        assert hasattr(fig, 'to_json')


# ===================== ADVANCED TUNING =====================

class TestOptunaOptimizer:

    def test_optimize_returns_fitted_model(self):
        from scomp_link.models.advanced_tuning import OptunaOptimizer
        np.random.seed(42)
        X = pd.DataFrame({'a': np.random.randn(100), 'b': np.random.randn(100)})
        y = 2 * X['a'] + np.random.randn(100) * 0.3

        def param_space(trial):
            return {'n_estimators': trial.suggest_int('n_estimators', 10, 50),
                    'max_depth': trial.suggest_int('max_depth', 2, 5)}

        optimizer = OptunaOptimizer(GradientBoostingRegressor, param_space,
                                    scoring='r2', n_trials=5, cv=3)
        best_model = optimizer.optimize(X, y, verbose=False)
        assert hasattr(best_model, 'predict')
        preds = best_model.predict(X)
        assert len(preds) == 100

    def test_best_params_before_optimize_raises(self):
        from scomp_link.models.advanced_tuning import OptunaOptimizer
        optimizer = OptunaOptimizer(LinearRegression, lambda t: {}, n_trials=5)
        with pytest.raises(ValueError):
            _ = optimizer.best_params

    def test_best_score_positive(self):
        from scomp_link.models.advanced_tuning import OptunaOptimizer
        np.random.seed(42)
        X = pd.DataFrame({'a': np.random.randn(100)})
        y = 3 * X['a'] + np.random.randn(100) * 0.1

        def param_space(trial):
            return {'fit_intercept': trial.suggest_categorical('fit_intercept', [True, False])}

        optimizer = OptunaOptimizer(LinearRegression, param_space, scoring='r2', n_trials=3, cv=3)
        optimizer.optimize(X, y, verbose=False)
        assert optimizer.best_score > 0.5


class TestHalvingSearchOptimizer:

    def test_optimize_returns_best_model(self):
        from scomp_link.models.advanced_tuning import HalvingSearchOptimizer
        np.random.seed(42)
        X = pd.DataFrame({'a': np.random.randn(200), 'b': np.random.randn(200)})
        y = X['a'] + np.random.randn(200) * 0.5

        optimizer = HalvingSearchOptimizer(
            GradientBoostingRegressor(),
            {'n_estimators': [10, 30, 50], 'max_depth': [2, 3, 5]},
            scoring='r2', cv=3
        )
        best_model = optimizer.optimize(X, y, verbose=False)
        assert hasattr(best_model, 'predict')
        assert optimizer.best_score > 0

    def test_results_dataframe(self):
        from scomp_link.models.advanced_tuning import HalvingSearchOptimizer
        np.random.seed(42)
        X = pd.DataFrame({'a': np.random.randn(200)})
        y = 2 * X['a']

        optimizer = HalvingSearchOptimizer(
            GradientBoostingRegressor(),
            {'n_estimators': [10, 30], 'max_depth': [2, 3]},
            scoring='r2', cv=2
        )
        optimizer.optimize(X, y, verbose=False)
        df = optimizer.results_dataframe()
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0


class TestEarlyStoppingCV:

    def test_finds_optimal_iterations(self):
        from scomp_link.models.advanced_tuning import EarlyStoppingCV
        np.random.seed(42)
        X = pd.DataFrame({'a': np.random.randn(150), 'b': np.random.randn(150)})
        y = X['a'] + np.random.randn(150) * 0.3

        stopper = EarlyStoppingCV(
            GradientBoostingRegressor(learning_rate=0.1),
            max_iterations=100, patience=30, step=10, cv=3
        )
        best_n, history = stopper.find_optimal_iterations(X, y)
        assert best_n >= 10
        assert isinstance(history, pd.DataFrame)
        assert 'mean_score' in history.columns

    def test_plot_learning_curve(self):
        from scomp_link.models.advanced_tuning import EarlyStoppingCV
        np.random.seed(42)
        X = pd.DataFrame({'a': np.random.randn(100)})
        y = X['a']

        stopper = EarlyStoppingCV(GradientBoostingRegressor(), max_iterations=30, step=10, cv=2)
        _, history = stopper.find_optimal_iterations(X, y)
        fig = stopper.plot_learning_curve(history)
        assert hasattr(fig, 'to_json')


# ===================== DRIFT DETECTION =====================

class TestDriftDetector:

    def test_detect_no_drift(self):
        from scomp_link import DriftDetector
        np.random.seed(42)
        ref = pd.DataFrame({'x': np.random.randn(500), 'y': np.random.randn(500)})
        curr = pd.DataFrame({'x': np.random.randn(500), 'y': np.random.randn(500)})
        detector = DriftDetector(ref)
        report = detector.detect(curr)
        assert report['drifted'].sum() == 0

    def test_detect_with_drift(self):
        from scomp_link import DriftDetector
        np.random.seed(42)
        ref = pd.DataFrame({'x': np.random.randn(500), 'y': np.random.randn(500)})
        curr = pd.DataFrame({'x': np.random.randn(500) + 5, 'y': np.random.randn(500)})
        detector = DriftDetector(ref)
        report = detector.detect(curr)
        assert report[report['feature'] == 'x']['drifted'].values[0] == True
        assert report[report['feature'] == 'y']['drifted'].values[0] == False

    def test_psi_threshold_respected(self):
        from scomp_link import DriftDetector
        np.random.seed(42)
        ref = pd.DataFrame({'x': np.random.randn(500)})
        curr = pd.DataFrame({'x': np.random.randn(500) + 10})
        detector = DriftDetector(ref, psi_threshold=0.1)
        report = detector.detect(curr)
        assert report.iloc[0]['psi'] > 0.1
        assert report.iloc[0]['psi_drifted'] == True

    def test_summary(self):
        from scomp_link import DriftDetector
        np.random.seed(42)
        ref = pd.DataFrame({'a': np.random.randn(300), 'b': np.random.randn(300)})
        curr = pd.DataFrame({'a': np.random.randn(300) + 3, 'b': np.random.randn(300)})
        detector = DriftDetector(ref)
        report = detector.detect(curr)
        summary = detector.summary(report)
        assert summary['total_features'] == 2
        assert summary['drifted_features'] >= 1
        assert summary['worst_feature'] == 'a'

    def test_plot_drift_report_returns_figure(self):
        from scomp_link import DriftDetector
        np.random.seed(42)
        ref = pd.DataFrame({'x': np.random.randn(200), 'y': np.random.randn(200)})
        curr = pd.DataFrame({'x': np.random.randn(200) + 2, 'y': np.random.randn(200)})
        detector = DriftDetector(ref)
        report = detector.detect(curr)
        fig = detector.plot_drift_report(report)
        assert hasattr(fig, 'to_json')

    def test_plot_feature_distribution(self):
        from scomp_link import DriftDetector
        np.random.seed(42)
        ref = pd.DataFrame({'x': np.random.randn(200)})
        curr = pd.DataFrame({'x': np.random.randn(200) + 3})
        detector = DriftDetector(ref)
        fig = detector.plot_feature_distribution('x', curr)
        assert hasattr(fig, 'to_json')


# ===================== PERSISTENCE =====================

class TestScompArtifact:

    def test_save_and_load(self, regression_model_and_data):
        from scomp_link import ScompArtifact
        model, X_train, X_test, _, _ = regression_model_and_data
        with tempfile.NamedTemporaryFile(suffix='.scomp', delete=False) as f:
            path = f.name
        try:
            artifact = ScompArtifact()
            artifact.set_model(model)
            artifact.set_config(task_type='regression', target_col='y')
            artifact.set_metrics({'r2': 0.95})
            artifact.set_feature_schema(X_train)
            artifact.set_sample_data(X_train)
            artifact.save(path)

            loaded = ScompArtifact.load(path)
            assert loaded.model is not None
            assert loaded.config['task_type'] == 'regression'
            assert loaded.metrics['r2'] == 0.95
            assert loaded.sample_data is not None
            assert len(loaded.feature_schema) == 3
        finally:
            os.unlink(path)

    def test_predict(self, regression_model_and_data):
        from scomp_link import ScompArtifact
        model, X_train, X_test, _, _ = regression_model_and_data
        with tempfile.NamedTemporaryFile(suffix='.scomp', delete=False) as f:
            path = f.name
        try:
            ScompArtifact().set_model(model).save(path)
            loaded = ScompArtifact.load(path)
            preds = loaded.predict(X_test)
            assert preds.shape == (60,)
        finally:
            os.unlink(path)

    def test_predict_without_model_raises(self):
        from scomp_link import ScompArtifact
        artifact = ScompArtifact()
        with pytest.raises(ValueError):
            artifact.predict(pd.DataFrame({'a': [1, 2]}))

    def test_is_scomp_file(self, regression_model_and_data):
        from scomp_link import ScompArtifact
        model, X_train, _, _, _ = regression_model_and_data
        with tempfile.NamedTemporaryFile(suffix='.scomp', delete=False) as f:
            path = f.name
        try:
            ScompArtifact().set_model(model).save(path)
            assert ScompArtifact.is_scomp_file(path) == True
            assert ScompArtifact.is_scomp_file('/nonexistent/file.scomp') == False
        finally:
            os.unlink(path)

    def test_info(self, regression_model_and_data):
        from scomp_link import ScompArtifact
        model, X_train, _, _, _ = regression_model_and_data
        artifact = ScompArtifact()
        artifact.set_model(model).set_config(task_type='regression')
        artifact.set_metrics({'rmse': 0.1}).set_feature_schema(X_train)
        info = artifact.info()
        assert info['model_type'] == 'RandomForestRegressor'
        assert info['has_model'] == True
        assert info['n_features'] == 3

    def test_metadata_persists(self, regression_model_and_data):
        from scomp_link import ScompArtifact
        model, _, _, _, _ = regression_model_and_data
        with tempfile.NamedTemporaryFile(suffix='.scomp', delete=False) as f:
            path = f.name
        try:
            ScompArtifact().set_model(model).set_metadata(author='test', experiment='v1').save(path)
            loaded = ScompArtifact.load(path)
            assert loaded.metadata['author'] == 'test'
            assert loaded.metadata['experiment'] == 'v1'
        finally:
            os.unlink(path)

    def test_auto_suffix(self, regression_model_and_data):
        from scomp_link import ScompArtifact
        model, _, _, _, _ = regression_model_and_data
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'my_pipeline')
            result_path = ScompArtifact().set_model(model).save(path)
            assert str(result_path).endswith('.scomp')
            assert os.path.exists(result_path)
