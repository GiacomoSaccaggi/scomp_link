# -*- coding: utf-8 -*-
"""
Dedicated tests for legacy modules: RegressorOptimizer, ClassifierOptimizer, ContrastiveTextClassifier
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# ===================== REGRESSOR OPTIMIZER =====================

class TestRegressorOptimizer:

    @pytest.fixture
    def regression_setup(self):
        np.random.seed(42)
        n = 300
        df = pd.DataFrame({
            'x1': np.random.randn(n),
            'x2': np.random.randn(n),
            'x3': np.random.randn(n),
            'cat': np.random.choice(['A', 'B', 'C'], n),
            'y': 2 * np.random.randn(n) + 0.5,
        })
        models = {
            'LinearRegression': {'model': LinearRegression(), 'params_grid': {'fit_intercept': [True]}},
            'Ridge': {'model': Ridge(), 'params_grid': {'alpha': [0.1, 1.0]}},
        }
        return df, models

    def test_initialization(self, regression_setup):
        from scomp_link import RegressorOptimizer
        df, models = regression_setup
        opt = RegressorOptimizer(df, 'y', ['x1', 'x2', 'x3'], 'x1', models_to_test=models)
        assert opt.y is not None
        assert opt.X_train is not None
        assert opt.X_test is not None

    def test_initialization_with_categorical(self, regression_setup):
        from scomp_link import RegressorOptimizer
        df, models = regression_setup
        opt = RegressorOptimizer(df, 'y', ['x1', 'x2', 'cat'], 'x1', models_to_test=models)
        assert 'cat' in opt.categorical_cols

    def test_test_models_regression(self, regression_setup):
        from scomp_link import RegressorOptimizer
        df, models = regression_setup
        opt = RegressorOptimizer(df, 'y', ['x1', 'x2'], 'x1', models_to_test=models)
        opt.test_models_regression()
        assert len(opt.model_results) == 2
        assert 'LinearRegression' in opt.model_results
        assert 'Ridge' in opt.model_results

    def test_model_results_structure(self, regression_setup):
        from scomp_link import RegressorOptimizer
        df, models = regression_setup
        opt = RegressorOptimizer(df, 'y', ['x1', 'x2'], 'x1', models_to_test=models)
        opt.test_models_regression()
        for name, result in opt.model_results.items():
            assert 'Params' in result
            assert 'Fitted_Test' in result
            assert 'True_Test' in result
            assert 'Model' in result

    def test_estimate_optimization_time(self, regression_setup):
        from scomp_link import RegressorOptimizer
        df, models = regression_setup
        opt = RegressorOptimizer(df, 'y', ['x1', 'x2'], 'x1', models_to_test=models)
        opt.estimate_optimization_time(time_per_combination=1)

    def test_preprocessor_exists(self, regression_setup):
        from scomp_link import RegressorOptimizer
        df, models = regression_setup
        opt = RegressorOptimizer(df, 'y', ['x1', 'x2'], 'x1', models_to_test=models)
        assert opt.preprocessor is not None


# ===================== CLASSIFIER OPTIMIZER =====================

class TestClassifierOptimizer:

    @pytest.fixture
    def classification_setup(self):
        np.random.seed(42)
        n = 200
        df = pd.DataFrame({
            'x1': np.random.randn(n),
            'x2': np.random.randn(n),
            'y': np.random.choice([0, 1, 2], n),
        })
        models = {
            'LogisticRegression': {'model': LogisticRegression(max_iter=200),
                                   'params_grid': {'C': [0.1, 1.0]}},
            'RandomForest': {'model': RandomForestClassifier(n_estimators=10),
                            'params_grid': {'max_depth': [3, 5]}},
        }
        return df, models

    def test_initialization(self, classification_setup):
        from scomp_link import ClassifierOptimizer
        df, models = classification_setup
        opt = ClassifierOptimizer(df, 'y', ['x1', 'x2'], models_to_test=models)
        assert opt.X_train is not None
        assert opt.y_train is not None

    def test_test_models_classification(self, classification_setup):
        from scomp_link import ClassifierOptimizer
        df, models = classification_setup
        opt = ClassifierOptimizer(df, 'y', ['x1', 'x2'], models_to_test=models)
        opt.test_models_classification()
        assert len(opt.model_results) == 2

    def test_results_have_model_and_params(self, classification_setup):
        from scomp_link import ClassifierOptimizer
        df, models = classification_setup
        opt = ClassifierOptimizer(df, 'y', ['x1', 'x2'], models_to_test=models)
        opt.test_models_classification()
        for name, result in opt.model_results.items():
            assert 'Model' in result
            assert 'Params' in result


# ===================== CONTRASTIVE TEXT CLASSIFIER =====================

class TestContrastiveTextClassifier:

    def test_import(self):
        from scomp_link.models.contrastive_text import ContrastiveTextClassifier
        assert ContrastiveTextClassifier is not None

    def test_initialization(self):
        from scomp_link.models.contrastive_text import ContrastiveTextClassifier
        clf = ContrastiveTextClassifier(model_name='bert-base-uncased')
        assert clf.use_faiss is True or clf.use_faiss is False
        assert hasattr(clf, 'labels')

    def test_has_required_methods(self):
        from scomp_link.models.contrastive_text import ContrastiveTextClassifier
        clf = ContrastiveTextClassifier(model_name='bert-base-uncased')
        assert hasattr(clf, 'train_contrastive')
        assert hasattr(clf, 'predict')
        assert hasattr(clf, 'predict_batch')
        assert hasattr(clf, 'save')
        assert hasattr(clf, 'load')

    def test_default_state(self):
        from scomp_link.models.contrastive_text import ContrastiveTextClassifier
        clf = ContrastiveTextClassifier(model_name='bert-base-uncased')
        assert clf.labels is None or clf.labels == []
        assert clf.faiss_index is None
