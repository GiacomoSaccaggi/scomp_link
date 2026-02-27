# -*- coding: utf-8 -*-
"""
Unified test suite for scomp_link package
"""
import pytest
import numpy as np
import pandas as pd
import os
import sys
from sklearn.datasets import make_classification, make_regression

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scomp_link import ScompLinkPipeline, Preprocessor, ModelFactory, Validator
from scomp_link.models.regressor_optimizer import RegressorOptimizer
from scomp_link.models.classifier_optimizer import ClassifierOptimizer


@pytest.fixture
def regression_data():
    X, y = make_regression(n_samples=500, n_features=5, noise=10, random_state=42)
    df = pd.DataFrame(X, columns=[f'x{i}' for i in range(5)])
    df['y'] = y
    return df


@pytest.fixture
def classification_data():
    X, y = make_classification(n_samples=500, n_features=5, n_classes=3, n_informative=3, random_state=42)
    df = pd.DataFrame(X, columns=[f'x{i}' for i in range(5)])
    df['y'] = y
    return df


@pytest.fixture
def small_data():
    return pd.DataFrame({
        'x1': np.random.randn(50),
        'x2': np.random.randn(50),
        'y': np.random.randn(50)
    })


class TestPipeline:
    
    def test_initialization(self):
        pipe = ScompLinkPipeline("Test")
        assert pipe.problem_description == "Test"
        assert pipe.objectives == []
    
    def test_set_objectives(self):
        pipe = ScompLinkPipeline("Test")
        pipe.set_objectives(["Minimize RMSE"])
        assert len(pipe.objectives) == 1
    
    def test_import_and_clean_data(self, small_data):
        pipe = ScompLinkPipeline("Test")
        pipe.import_and_clean_data(small_data)
        assert pipe.df is not None
    
    def test_select_variables(self, small_data):
        pipe = ScompLinkPipeline("Test")
        pipe.import_and_clean_data(small_data)
        pipe.select_variables(target_col='y')
        assert pipe.target_col == 'y'
        assert 'y' not in pipe.feature_cols
    
    def test_choose_model_numerical(self, small_data):
        pipe = ScompLinkPipeline("Test")
        pipe.import_and_clean_data(small_data)
        pipe.select_variables(target_col='y')
        pipe.choose_model("numerical_prediction")
        assert pipe.model_type == "Econometric Model"
    
    def test_run_pipeline_regression(self, small_data):
        pipe = ScompLinkPipeline("Test")
        pipe.set_objectives(["Minimize RMSE"])
        pipe.import_and_clean_data(small_data)
        pipe.select_variables(target_col='y')
        pipe.choose_model("numerical_prediction")
        results = pipe.run_pipeline(task_type="regression")
        assert results["status"] == "success"
        assert "metrics" in results
    
    def test_run_pipeline_without_data_raises_error(self):
        pipe = ScompLinkPipeline("Test")
        with pytest.raises(ValueError, match="Data and target must be defined"):
            pipe.run_pipeline()


class TestPreprocessor:
    
    def test_initialization(self, small_data):
        prep = Preprocessor(small_data)
        assert prep.df is not None
    
    def test_clean_data_removes_duplicates(self):
        df = pd.DataFrame({'x': [1, 1, 2], 'y': [1, 1, 2]})
        prep = Preprocessor(df)
        cleaned = prep.clean_data(remove_outliers=False)
        assert len(cleaned) == 2
    
    def test_integrate_data(self):
        df1 = pd.DataFrame({'id': [1, 2], 'x': [10, 20]})
        df2 = pd.DataFrame({'id': [1, 2], 'y': [100, 200]})
        prep = Preprocessor(df1)
        integrated = prep.integrate_data(df2, on='id')
        assert 'y' in integrated.columns
    
    def test_run_eda(self, small_data):
        prep = Preprocessor(small_data)
        summary = prep.run_eda()
        assert "shape" in summary
        assert "missing_values" in summary
    
    def test_prepare_datasets(self, small_data):
        prep = Preprocessor(small_data)
        X_train, X_test, y_train, y_test = prep.prepare_datasets('y', test_size=0.2)
        assert len(X_train) > len(X_test)


class TestModelFactory:
    
    def test_get_ridge_model(self):
        model = ModelFactory.get_model("Ridge")
        from sklearn.linear_model import Ridge
        assert isinstance(model, Ridge)
    
    def test_get_lasso_model(self):
        model = ModelFactory.get_model("Lasso / Elastic Net")
        assert model is not None
    
    def test_get_random_forest(self):
        model = ModelFactory.get_model("Random Forest")
        assert model is not None
    
    def test_get_kmeans(self):
        model = ModelFactory.get_model("KMeans", n_clusters=3)
        assert model is not None
    
    def test_unknown_model_returns_none(self):
        model = ModelFactory.get_model("UnknownModel")
        assert model is None


class TestValidator:
    
    def test_initialization(self):
        from sklearn.linear_model import LinearRegression
        model = LinearRegression()
        validator = Validator(model)
        assert validator.model is not None
    
    def test_evaluate_regression(self, regression_data):
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split
        
        X = regression_data.drop('y', axis=1)
        y = regression_data['y']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        validator = Validator(model)
        metrics = validator.evaluate(y_test, y_pred, task_type="regression")
        
        assert "mse" in metrics
        assert "rmse" in metrics
        assert "r2" in metrics
    
    def test_k_fold_cv(self, small_data):
        from sklearn.linear_model import LinearRegression
        
        X = small_data.drop('y', axis=1)
        y = small_data['y']
        
        model = LinearRegression()
        validator = Validator(model)
        scores = validator.k_fold_cv(X, y, k=3)
        
        assert len(scores) == 3


class TestIntegration:
    
    def test_full_regression_pipeline(self, regression_data):
        pipe = ScompLinkPipeline("Full Test")
        pipe.set_objectives(["Minimize RMSE"])
        pipe.import_and_clean_data(regression_data)
        pipe.select_variables(target_col='y')
        pipe.choose_model("numerical_prediction", metadata={"only_numerical_exogenous": True, "all_variables_important": False})
        results = pipe.run_pipeline(task_type="regression")
        
        assert results["status"] == "success"
        assert results["metrics"]["r2"] is not None
    
    def test_full_classification_pipeline(self, classification_data):
        from sklearn.ensemble import RandomForestClassifier
        
        pipe = ScompLinkPipeline("Classification Test")
        pipe.set_objectives(["Maximize Accuracy"])
        pipe.import_and_clean_data(classification_data)
        pipe.select_variables(target_col='y')
        pipe.choose_model("categorical_known", metadata={"records_per_category": 500})
        pipe.model = RandomForestClassifier(random_state=42)
        results = pipe.run_pipeline(task_type="classification")
        
        assert results["status"] == "success"
        assert results["metrics"]["accuracy"] > 0


class TestImageModels:
    
    def test_cnn_img_initialization(self):
        from scomp_link.models.supervised_img import CNNImg
        model = CNNImg(n_classes=3)
        assert model.n_classes == 3
    
    def test_cnn_img_fit_predict(self):
        from scomp_link.models.supervised_img import CNNImg
        
        X = np.random.rand(10, 32, 32, 3)
        y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
        
        model = CNNImg()
        model.fit(X, y)
        
        assert model.classes_ is not None
        predictions = model.predict(X)
        assert len(predictions) == len(X)
    
    def test_cluster_img_initialization(self):
        from scomp_link.models.unsupervised_img import ClusterImg
        model = ClusterImg(n_clusters=5)
        assert model.n_clusters == 5
    
    def test_cluster_img_fit_predict(self):
        from scomp_link.models.unsupervised_img import ClusterImg
        
        X = np.random.rand(20, 32, 32, 3)
        model = ClusterImg(n_clusters=3)
        labels = model.fit_predict(X)
        
        assert len(labels) == len(X)
        assert len(np.unique(labels)) <= 3


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--cov=scomp_link", "--cov-report=html", "--cov-report=term"])
