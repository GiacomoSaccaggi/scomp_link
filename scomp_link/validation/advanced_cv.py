# -*- coding: utf-8 -*-
"""

 ██████╗ ██████╗ ██╗   ██╗ ██████╗ ██╗   ██╗ ██████╗███████╗██████╗     ██████╗██╗   ██╗
██╔════╝██╔════╝██║   ██║██╔════╝██║   ██║██╔════╝██╔════╝██╔══██╗    ██╔════╝██║   ██║
█████╗  ██║  ██╗╚██╗ ██╔╝█████╗  ██║   ██║█████╗  █████╗  ██║  ██║    ██║     ╚██╗ ██╔╝
██╔══╝  ██║   ██║ ╚████╔╝ ██╔══╝  ╚██╗ ██╔╝██╔══╝  ██╔══╝  ██║  ██║    ██║      ╚████╔╝ 
███████╗╚██████╔╝  ╚██╔╝  ███████╗ ╚████╔╝ ███████╗███████╗╚██████╔╝    ╚███████╗ ╚██╔╝  
╚══════╝ ╚═════╝    ╚═╝   ╚══════╝  ╚═══╝  ╚══════╝╚══════╝ ╚═════╝     ╚══════╝  ╚═╝   

Advanced Cross-Validation strategies including LOOCV and Bootstrap
"""
import numpy as np
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.utils import resample
from sklearn.base import clone


class AdvancedCV:
    """
    AdvancedCV provides advanced cross-validation strategies beyond standard K-Fold.
    
    Dependencies:
    - numpy
    - sklearn.model_selection
    - sklearn.utils
    
    Methods:
    - loocv: Leave-One-Out Cross Validation
    - bootstrap: Bootstrap resampling validation
    - evaluate_all: Run all validation strategies
    """
    
    @staticmethod
    def loocv(model, X, y, scoring=None):
        """
        Leave-One-Out Cross Validation
        
        Parameters:
        - model: Estimator to validate
        - X: Features
        - y: Target
        - scoring: Scoring metric (default: accuracy for classification, r2 for regression)
        
        Returns:
        - dict with mean, std, and all scores
        """
        loo = LeaveOneOut()
        scores = cross_val_score(model, X, y, cv=loo, scoring=scoring)
        
        return {
            'method': 'LOOCV',
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'n_splits': len(scores),
            'scores': scores
        }
    
    @staticmethod
    def bootstrap(model, X, y, n_iterations=1000, test_size=0.3, random_state=42):
        """
        Bootstrap validation with out-of-bag evaluation
        
        Parameters:
        - model: Estimator to validate
        - X: Features
        - y: Target
        - n_iterations: Number of bootstrap samples
        - test_size: Proportion of data for testing
        - random_state: Random seed
        
        Returns:
        - dict with mean, std, and all scores
        """
        np.random.seed(random_state)
        n_samples = len(X)
        n_test = int(n_samples * test_size)
        scores = []
        
        for i in range(n_iterations):
            # Bootstrap sample
            train_idx = resample(range(n_samples), n_samples=n_samples - n_test, random_state=random_state + i)
            test_idx = list(set(range(n_samples)) - set(train_idx))
            
            if len(test_idx) == 0:
                continue
            
            X_train = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
            y_train = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
            X_test = X.iloc[test_idx] if hasattr(X, 'iloc') else X[test_idx]
            y_test = y.iloc[test_idx] if hasattr(y, 'iloc') else y[test_idx]
            
            # Clone and fit model
            model_clone = clone(model)
            model_clone.fit(X_train, y_train)
            score = model_clone.score(X_test, y_test)
            scores.append(score)
        
        scores = np.array(scores)
        
        return {
            'method': 'Bootstrap',
            'mean_score': scores.mean(),
            'std_score': scores.std(),
            'n_iterations': len(scores),
            'scores': scores,
            'confidence_interval_95': (np.percentile(scores, 2.5), np.percentile(scores, 97.5))
        }
    
    @staticmethod
    def evaluate_all(model, X, y, include_loocv=True, include_bootstrap=True, 
                     bootstrap_iterations=1000, scoring=None):
        """
        Run all advanced validation strategies
        
        Parameters:
        - model: Estimator to validate
        - X: Features
        - y: Target
        - include_loocv: Whether to run LOOCV (can be slow for large datasets)
        - include_bootstrap: Whether to run Bootstrap
        - bootstrap_iterations: Number of bootstrap samples
        - scoring: Scoring metric
        
        Returns:
        - dict with results from all methods
        """
        results = {}
        
        # LOOCV - skip if dataset is too large
        if include_loocv and len(X) < 1000:
            print("Running Leave-One-Out Cross Validation...")
            results['loocv'] = AdvancedCV.loocv(model, X, y, scoring=scoring)
        elif include_loocv:
            print("⚠️  LOOCV skipped: dataset too large (>1000 samples)")
        
        # Bootstrap
        if include_bootstrap:
            print(f"Running Bootstrap validation ({bootstrap_iterations} iterations)...")
            results['bootstrap'] = AdvancedCV.bootstrap(model, X, y, n_iterations=bootstrap_iterations)
        
        return results
