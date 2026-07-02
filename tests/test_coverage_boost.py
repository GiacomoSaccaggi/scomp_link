# -*- coding: utf-8 -*-
"""
Tests targeting low-coverage modules: core.py, regressor_optimizer, decorators, highcharts
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor


# ===================== CORE.PY (57%) =====================

class TestCorePipeline:

    @pytest.fixture
    def medium_data(self):
        np.random.seed(42)
        n = 200
        df = pd.DataFrame({
            'x1': np.random.randn(n),
            'x2': np.random.randn(n),
            'x3': np.random.randn(n),
            'y': 2 * np.random.randn(n) + 0.5,
        })
        return df

    @pytest.fixture
    def classification_data(self):
        np.random.seed(42)
        n = 200
        df = pd.DataFrame({
            'x1': np.random.randn(n),
            'x2': np.random.randn(n),
            'y': np.random.choice([0, 1], n),
        })
        return df

    def test_full_regression_with_ensemble(self, medium_data):
        from scomp_link import ScompLinkPipeline
        pipe = ScompLinkPipeline("Ensemble Test")
        pipe.set_objectives(["Minimize RMSE"])
        pipe.import_and_clean_data(medium_data)
        pipe.select_variables(target_col='y')
        pipe.choose_model("numerical_prediction")
        results = pipe.run_pipeline(task_type="regression", use_ensemble=True, test_size=0.3)
        assert results['status'] == 'success'

    def test_full_regression_with_advanced_cv(self, medium_data):
        from scomp_link import ScompLinkPipeline
        pipe = ScompLinkPipeline("CV Test")
        pipe.set_objectives(["Minimize RMSE"])
        pipe.import_and_clean_data(medium_data)
        pipe.select_variables(target_col='y')
        pipe.choose_model("numerical_prediction")
        results = pipe.run_pipeline(task_type="regression", advanced_cv=True, test_size=0.3)
        assert results['status'] == 'success'

    def test_classification_pipeline(self, classification_data):
        from scomp_link import ScompLinkPipeline
        pipe = ScompLinkPipeline("Classification Test")
        pipe.set_objectives(["Maximize Accuracy"])
        pipe.import_and_clean_data(classification_data)
        pipe.select_variables(target_col='y')
        pipe.choose_model("categorical_known")
        results = pipe.run_pipeline(task_type="classification", test_size=0.3)
        assert results['status'] == 'success'

    def test_save_and_load_model(self, medium_data):
        import tempfile, os
        from scomp_link import ScompLinkPipeline
        pipe = ScompLinkPipeline("Save Test")
        pipe.import_and_clean_data(medium_data)
        pipe.select_variables(target_col='y')
        pipe.choose_model("numerical_prediction")
        pipe.run_pipeline(task_type="regression")

        with tempfile.TemporaryDirectory() as d:
            pipe.save_model(d)
            pipe2 = ScompLinkPipeline("Load Test")
            pipe2.load_model(d)
            assert pipe2.model is not None

    def test_predict_after_training(self, medium_data):
        from scomp_link import ScompLinkPipeline
        pipe = ScompLinkPipeline("Predict Test")
        pipe.import_and_clean_data(medium_data)
        pipe.select_variables(target_col='y')
        pipe.choose_model("numerical_prediction")
        pipe.run_pipeline(task_type="regression")
        preds = pipe.predict(medium_data[['x1', 'x2', 'x3']].head(10))
        assert len(preds) == 10


# ===================== REGRESSOR OPTIMIZER (24%) =====================

class TestRegressorOptimizerDetailed:

    @pytest.fixture
    def setup(self):
        np.random.seed(42)
        n = 200
        df = pd.DataFrame({
            'x1': np.random.randn(n),
            'x2': np.random.randn(n),
            'y': 3 * np.random.randn(n) + 1,
        })
        models = {
            'LinearRegression': {'model': LinearRegression(), 'params_grid': {'fit_intercept': [True, False]}},
            'GBR': {'model': GradientBoostingRegressor(n_estimators=10),
                    'params_grid': {'max_depth': [2, 3]}},
        }
        return df, models

    def test_full_optimization_flow(self, setup):
        from scomp_link import RegressorOptimizer
        df, models = setup
        opt = RegressorOptimizer(df, 'y', ['x1', 'x2'], 'x1', models_to_test=models)
        opt.test_models_regression()
        assert 'LinearRegression' in opt.model_results
        assert 'GBR' in opt.model_results
        # Check fitted models exist
        for name, result in opt.model_results.items():
            assert result['Model'] is not None
            assert 'Fitted_Test' in result
            assert len(result['Fitted_Test']) > 0

    def test_train_test_split_correct(self, setup):
        from scomp_link import RegressorOptimizer
        df, models = setup
        opt = RegressorOptimizer(df, 'y', ['x1', 'x2'], 'x1', models_to_test=models)
        total = len(opt.X_train) + len(opt.X_test)
        assert total == len(df)

    def test_preprocessor_transforms(self, setup):
        from scomp_link import RegressorOptimizer
        df, models = setup
        opt = RegressorOptimizer(df, 'y', ['x1', 'x2'], 'x1', models_to_test=models)
        # Preprocessor should work without error
        X_transformed = opt.preprocessor.fit_transform(opt.X_train)
        assert X_transformed.shape[0] == len(opt.X_train)

    def test_multiple_models_produces_results(self, setup):
        from scomp_link import RegressorOptimizer
        df, models = setup
        opt = RegressorOptimizer(df, 'y', ['x1', 'x2'], 'x1', models_to_test=models)
        opt.test_models_regression()
        # Both models should have predictions
        for name in models:
            assert name in opt.model_results
            preds = opt.model_results[name]['Fitted_Test']
            assert len(preds) == len(opt.y_test)


# ===================== DECORATORS (31%) =====================

class TestDecorators:

    def test_timer(self):
        from scomp_link.utils.decorators import timer

        @timer
        def slow_func():
            return 42

        result = slow_func()
        assert result == 42

    def test_retry_success(self):
        from scomp_link.utils.decorators import retry
        call_count = [0]

        @retry(max_attempts=3, delay=0)
        def flaky():
            call_count[0] += 1
            if call_count[0] < 3:
                raise ValueError("not yet")
            return "ok"

        assert flaky() == "ok"
        assert call_count[0] == 3

    def test_retry_exhausted(self):
        from scomp_link.utils.decorators import retry

        @retry(max_attempts=2, delay=0)
        def always_fails():
            raise RuntimeError("fail")

        with pytest.raises(RuntimeError):
            always_fails()

    def test_cache(self):
        from scomp_link.utils.decorators import cache
        call_count = [0]

        @cache
        def expensive(n):
            call_count[0] += 1
            return n * 2

        assert expensive(5) == 10
        assert expensive(5) == 10  # cached
        assert call_count[0] == 1
        expensive.cache_clear()
        expensive(5)
        assert call_count[0] == 2

    def test_deprecated(self):
        from scomp_link.utils.decorators import deprecated
        import warnings

        @deprecated("Use new_func instead")
        def old_func():
            return 1

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = old_func()
            assert result == 1
            assert len(w) == 1
            assert "deprecated" in str(w[0].message)

    def test_suppress_exceptions(self):
        from scomp_link.utils.decorators import suppress_exceptions

        @suppress_exceptions(default=-1, log=False)
        def crasher():
            raise ValueError("boom")

        assert crasher() == -1

    def test_validate_args(self):
        from scomp_link.utils.decorators import validate_args

        @validate_args(x=lambda v: v > 0, name=lambda v: len(v) > 0)
        def func(x, name="test"):
            return x

        assert func(5) == 5
        with pytest.raises(ValueError):
            func(-1)
        with pytest.raises(ValueError):
            func(1, name="")

    def test_run_once(self):
        from scomp_link.utils.decorators import run_once
        call_count = [0]

        @run_once
        def init():
            call_count[0] += 1
            return "initialized"

        assert init() == "initialized"
        assert init() == "initialized"
        assert call_count[0] == 1

    def test_log_call(self):
        from scomp_link.utils.decorators import log_call

        @log_call
        def add(a, b):
            return a + b

        result = add(3, 4)
        assert result == 7

    def test_memory_usage(self):
        from scomp_link.utils.decorators import memory_usage

        @memory_usage
        def alloc():
            return [0] * 10000

        result = alloc()
        assert len(result) == 10000


# ===================== HIGHCHARTS (0%) =====================

class TestHighcharts:

    def test_streamgraphs(self):
        from scomp_link.utils.highcharts import streamgraphs
        dates = ["2024-01", "2024-02", "2024-03"]
        series = {"A": [10, 20, 30], "B": [5, 15, 25]}
        html = streamgraphs("Test", dates, series)
        assert "Highcharts" in html
        assert "Test" in html

    def test_streamgraphs_area(self):
        from scomp_link.utils.highcharts import streamgraphs
        dates = ["2024-01", "2024-02"]
        series = {"X": [1, 2]}
        html = streamgraphs("Area", dates, series, area=True)
        assert "Highcharts" in html

    def test_calendar_heatmap(self):
        from scomp_link.utils.highcharts import calendar_heatmap
        series = {"2024-01-01": 50, "2024-01-02": 80, "2024-01-03": 30}
        html = calendar_heatmap("Heatmap", series, min=0, max=100)
        assert "Heatmap" in html
        assert "Highcharts" in html

    def test_calendar_gantt(self):
        from scomp_link.utils.highcharts import calendar_gantt
        # Just verify function exists and is callable
        assert callable(calendar_gantt)


# ===================== CORE.PY — choose_model branches =====================

class TestChooseModelBranches:
    """Exercise all choose_model branches in core.py."""

    @pytest.fixture
    def pipeline_with_data(self):
        np.random.seed(42)
        n = 200
        df = pd.DataFrame({
            'x1': np.random.randn(n),
            'x2': np.random.randn(n),
            'y': np.random.randn(n),
        })
        from scomp_link import ScompLinkPipeline
        pipe = ScompLinkPipeline("Branch Test")
        pipe.import_and_clean_data(df)
        pipe.select_variables(target_col='y')
        return pipe

    def test_categorical_known_image_pretrained(self, pipeline_with_data):
        """Lines 71-72: data_type=images, count_per_category < 500."""
        pipe = pipeline_with_data
        pipe.choose_model("categorical_known", metadata={"data_type": "images", "count_per_category": 100})
        assert pipe.model_type == "Pre-trained model"

    def test_categorical_known_image_cnn(self, pipeline_with_data):
        """Lines 73-74: data_type=images, count_per_category >= 500."""
        pipe = pipeline_with_data
        pipe.choose_model("categorical_known", metadata={"data_type": "images", "count_per_category": 1000})
        assert pipe.model_type == "CNN (ResNet/Inception)"

    def test_categorical_known_psychometric(self, pipeline_with_data):
        """Lines 76-77: exogenous_type=categorical, num_features < 5."""
        pipe = pipeline_with_data
        pipe.choose_model("categorical_known", metadata={"exogenous_type": "categorical", "num_features": 3})
        assert pipe.model_type == "Theorical Psychometric Model"

    def test_categorical_known_naive_bayes(self, pipeline_with_data):
        """Lines 78-79: exogenous_type=categorical, num_features >= 5."""
        pipe = pipeline_with_data
        pipe.choose_model("categorical_known", metadata={"exogenous_type": "categorical", "num_features": 10})
        assert pipe.model_type == "Naive Bayes / Classification Tree"

    def test_numerical_study_geospatial(self, pipeline_with_data):
        """Lines 93-94: numerical_study + geospatial."""
        pipe = pipeline_with_data
        pipe.choose_model("numerical_study", metadata={"geospatial": True})
        assert pipe.model_type == "Geostatistical Model / Kriging"

    def test_numerical_study_time_series(self, pipeline_with_data):
        """Lines 95-96: numerical_study + time_series."""
        pipe = pipeline_with_data
        pipe.choose_model("numerical_study", metadata={"time_series": True})
        assert pipe.model_type == "UCM State Space"

    def test_numerical_study_pca(self, pipeline_with_data):
        """Lines 97-98: numerical_study default."""
        pipe = pipeline_with_data
        pipe.choose_model("numerical_study", metadata={})
        assert pipe.model_type == "Randomized PCA / Statistical Tests"

    def test_numerical_prediction_econometric(self):
        """Line 107: numerical_prediction with < 1000 records."""
        np.random.seed(42)
        df = pd.DataFrame({'x': np.random.randn(500), 'y': np.random.randn(500)})
        from scomp_link import ScompLinkPipeline
        pipe = ScompLinkPipeline("Small Data")
        pipe.import_and_clean_data(df)
        pipe.select_variables(target_col='y')
        pipe.choose_model("numerical_prediction")
        assert pipe.model_type == "Econometric Model"

    def test_multi_numerical_prediction_var(self, pipeline_with_data):
        """Lines 116-117: multi_numerical_prediction + time_series."""
        pipe = pipeline_with_data
        pipe.choose_model("multi_numerical_prediction", metadata={"time_series": True})
        assert pipe.model_type == "VAR / VARMA"

    def test_multi_numerical_prediction_mlp(self, pipeline_with_data):
        """Lines 118-119: multi_numerical_prediction default."""
        pipe = pipeline_with_data
        pipe.choose_model("multi_numerical_prediction", metadata={})
        assert pipe.model_type == "MLP"
