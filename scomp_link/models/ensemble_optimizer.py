# -*- coding: utf-8 -*-
"""

███████╗███╗  ██╗ ██████╗███████╗███╗   ███╗██████╗ ██╗     ███████╗
██╔════╝████╗ ██║██╔════╝██╔════╝████╗ ████║██╔══██╗██║     ██╔════╝
█████╗  ██╔██╗██║╚█████╗ █████╗  ██╔████╔██║██████╦╝██║     █████╗  
██╔══╝  ██║╚████║ ╚═══██╗██╔══╝  ██║╚██╔╝██║██╔══██╗██║     ██╔══╝  
███████╗██║ ╚███║██████╔╝███████╗██║ ╚═╝ ██║██████╦╝███████╗███████╗
╚══════╝╚═╝  ╚══╝╚═════╝ ╚══════╝╚═╝     ╚═╝╚═════╝ ╚══════╝╚══════╝

██╗     ███████╗ █████╗ ██████╗ ███╗  ██╗██╗███╗  ██╗ ██████╗ 
██║     ██╔════╝██╔══██╗██╔══██╗████╗ ██║██║████╗ ██║██╔════╝ 
██║     █████╗  ███████║██████╔╝██╔██╗██║██║██╔██╗██║██║  ██╗ 
██║     ██╔══╝  ██╔══██║██╔══██╗██║╚████║██║██║╚████║██║  ╚██╗
███████╗███████╗██║  ██║██║  ██║██║ ╚███║██║██║ ╚███║╚██████╔╝
╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚══╝╚═╝╚═╝  ╚══╝ ╚═════╝ 
"""
import numpy as np
from sklearn.ensemble import VotingClassifier, VotingRegressor, StackingClassifier, StackingRegressor
from sklearn.model_selection import cross_val_score


class EnsembleOptimizer:
    """
    EnsembleOptimizer combines multiple trained models using voting or stacking strategies.
    
    Dependencies:
    - numpy
    - sklearn.ensemble
    
    Parameters:
    - base_models: List of tuples (name, model) with trained models
    - task_type: 'classification' or 'regression'
    - strategy: 'voting' or 'stacking'
    
    Methods:
    - create_ensemble: Creates ensemble model
    - evaluate_ensemble: Evaluates ensemble performance
    """
    
    def __init__(self, base_models, task_type='classification', strategy='voting'):
        self.base_models = base_models
        self.task_type = task_type
        self.strategy = strategy
        self.ensemble_model = None
    
    def create_ensemble(self, meta_model=None):
        """
        Create ensemble model using voting or stacking
        
        Parameters:
        - meta_model: Meta-learner for stacking (optional)
        
        Returns:
        - ensemble_model: Fitted ensemble model
        """
        if self.strategy == 'voting':
            if self.task_type == 'classification':
                self.ensemble_model = VotingClassifier(
                    estimators=self.base_models,
                    voting='soft'
                )
            else:
                self.ensemble_model = VotingRegressor(
                    estimators=self.base_models
                )
        
        elif self.strategy == 'stacking':
            if self.task_type == 'classification':
                self.ensemble_model = StackingClassifier(
                    estimators=self.base_models,
                    final_estimator=meta_model,
                    cv=5
                )
            else:
                self.ensemble_model = StackingRegressor(
                    estimators=self.base_models,
                    final_estimator=meta_model,
                    cv=5
                )
        
        return self.ensemble_model
    
    def fit(self, X, y):
        """Fit ensemble model"""
        if self.ensemble_model is None:
            self.create_ensemble()
        self.ensemble_model.fit(X, y)
        return self
    
    def predict(self, X):
        """Make predictions with ensemble"""
        return self.ensemble_model.predict(X)
    
    def evaluate_ensemble(self, X, y, cv=5):
        """
        Evaluate ensemble using cross-validation
        
        Parameters:
        - X: Features
        - y: Target
        - cv: Number of folds
        
        Returns:
        - scores: Cross-validation scores
        """
        if self.ensemble_model is None:
            self.create_ensemble()
        
        scoring = 'accuracy' if self.task_type == 'classification' else 'r2'
        scores = cross_val_score(self.ensemble_model, X, y, cv=cv, scoring=scoring)
        
        return {
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'scores': scores
        }
