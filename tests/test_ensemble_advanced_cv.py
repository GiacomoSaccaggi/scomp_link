# -*- coding: utf-8 -*-
"""
Tests for Ensemble Learning and Advanced Cross-Validation
"""
import pytest
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression, make_classification
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

from scomp_link.models.ensemble_optimizer import EnsembleOptimizer
from scomp_link.validation.advanced_cv import AdvancedCV


@pytest.fixture
def regression_data():
    X, y = make_regression(n_samples=200, n_features=5, noise=10, random_state=42)
    df = pd.DataFrame(X, columns=[f'x{i}' for i in range(5)])
    df['y'] = y
    return df


@pytest.fixture
def classification_data():
    X, y = make_classification(n_samples=200, n_features=5, n_classes=2, random_state=42)
    df = pd.DataFrame(X, columns=[f'x{i}' for i in range(5)])
    df['y'] = y
    return df


class TestEnsembleOptimizer:
    
    def test_voting_regressor_initialization(self):
        base_models = [
            ('lr', LinearRegression()),
            ('ridge', Ridge())
        ]
        ensemble = EnsembleOptimizer(base_models, task_type='regression', strategy='voting')
        assert ensemble.task_type == 'regression'
        assert ensemble.strategy == 'voting'
    
    def test_voting_classifier_initialization(self):
        base_models = [
            ('lr', LogisticRegression()),
            ('rf', RandomForestClassifier(n_estimators=10, random_state=42))
        ]
        ensemble = EnsembleOptimizer(base_models, task_type='classification', strategy='voting')
        assert ensemble.task_type == 'classification'
    
    def test_create_voting_ensemble_regression(self, regression_data):
        X = regression_data.drop('y', axis=1)
        y = regression_data['y']
        
        base_models = [
            ('lr', LinearRegression()),
            ('ridge', Ridge())
        ]
        ensemble = EnsembleOptimizer(base_models, task_type='regression', strategy='voting')
        ensemble_model = ensemble.create_ensemble()
        
        assert ensemble_model is not None
        ensemble_model.fit(X, y)
        predictions = ensemble_model.predict(X)
        assert len(predictions) == len(y)
    
    def test_create_stacking_ensemble_regression(self, regression_data):
        X = regression_data.drop('y', axis=1)
        y = regression_data['y']
        
        base_models = [
            ('lr', LinearRegression()),
            ('ridge', Ridge())
        ]
        ensemble = EnsembleOptimizer(base_models, task_type='regression', strategy='stacking')
        ensemble_model = ensemble.create_ensemble(meta_model=Ridge())
        
        assert ensemble_model is not None
        ensemble_model.fit(X, y)
        predictions = ensemble_model.predict(X)
        assert len(predictions) == len(y)
    
    def test_fit_predict(self, regression_data):
        X = regression_data.drop('y', axis=1)
        y = regression_data['y']
        
        base_models = [
            ('lr', LinearRegression()),
            ('ridge', Ridge())
        ]
        ensemble = EnsembleOptimizer(base_models, task_type='regression', strategy='voting')
        ensemble.fit(X, y)
        predictions = ensemble.predict(X)
        
        assert len(predictions) == len(y)
    
    def test_evaluate_ensemble(self, regression_data):
        X = regression_data.drop('y', axis=1)
        y = regression_data['y']
        
        base_models = [
            ('lr', LinearRegression()),
            ('ridge', Ridge())
        ]
        ensemble = EnsembleOptimizer(base_models, task_type='regression', strategy='voting')
        scores = ensemble.evaluate_ensemble(X, y, cv=3)
        
        assert 'mean_score' in scores
        assert 'std_score' in scores
        assert 'scores' in scores


class TestAdvancedCV:
    
    def test_loocv_regression(self, regression_data):
        X = regression_data.drop('y', axis=1)
        y = regression_data['y']
        
        model = LinearRegression()
        result = AdvancedCV.loocv(model, X, y)
        
        assert result['method'] == 'LOOCV'
        assert 'mean_score' in result
        assert 'std_score' in result
        assert result['n_splits'] == len(X)
    
    def test_bootstrap_regression(self, regression_data):
        X = regression_data.drop('y', axis=1)
        y = regression_data['y']
        
        model = LinearRegression()
        result = AdvancedCV.bootstrap(model, X, y, n_iterations=100)
        
        assert result['method'] == 'Bootstrap'
        assert 'mean_score' in result
        assert 'std_score' in result
        assert 'confidence_interval_95' in result
        assert result['n_iterations'] == 100
    
    def test_bootstrap_classification(self, classification_data):
        X = classification_data.drop('y', axis=1)
        y = classification_data['y']
        
        model = LogisticRegression(max_iter=200)
        result = AdvancedCV.bootstrap(model, X, y, n_iterations=50)
        
        assert result['method'] == 'Bootstrap'
        assert result['n_iterations'] == 50
    
    def test_evaluate_all_small_dataset(self, regression_data):
        X = regression_data.drop('y', axis=1)
        y = regression_data['y']
        
        model = LinearRegression()
        results = AdvancedCV.evaluate_all(
            model, X, y,
            include_loocv=True,
            include_bootstrap=True,
            bootstrap_iterations=50
        )
        
        assert 'loocv' in results
        assert 'bootstrap' in results
    
    def test_evaluate_all_skip_loocv_large_dataset(self):
        # Create large dataset
        X, y = make_regression(n_samples=1500, n_features=5, random_state=42)
        X = pd.DataFrame(X)
        y = pd.Series(y)
        
        model = LinearRegression()
        results = AdvancedCV.evaluate_all(
            model, X, y,
            include_loocv=True,
            include_bootstrap=True,
            bootstrap_iterations=50
        )
        
        # LOOCV should be skipped for large datasets
        assert 'loocv' not in results
        assert 'bootstrap' in results


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
