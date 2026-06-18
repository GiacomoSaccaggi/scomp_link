# -*- coding: utf-8 -*-
"""
████████╗██╗   ██╗███╗   ██╗██╗███╗   ██╗ ██████╗ 
╚══██╔══╝██║   ██║████╗  ██║██║████╗  ██║██╔════╝ 
   ██║   ██║   ██║██╔██╗ ██║██║██╔██╗ ██║██║  ██╗ 
   ██║   ██║   ██║██║╚████║██║██║╚████║██║  ╚██╗
   ██║   ╚██████╔╝██║ ╚███║██║██║ ╚███║╚██████╔╝
   ╚═╝    ╚═════╝ ╚═╝  ╚══╝╚═╝╚═╝  ╚══╝ ╚═════╝ 
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Callable, List, Union
from sklearn.model_selection import cross_val_score

from scomp_link.utils.logger import get_logger
logger = get_logger(__name__)
from scomp_link.utils.decorators import timer



class OptunaOptimizer:
    """
    Bayesian hyperparameter optimization using Optuna.
    
    Dependencies: optuna, scikit-learn
    
    PARAMETERS:
     1. estimator_class: sklearn estimator class (not instance)
     2. param_space: dict mapping param names to optuna suggest callables
     3. scoring: sklearn scoring string (e.g. 'neg_mean_squared_error', 'accuracy')
     4. cv: number of cross-validation folds
     5. n_trials: number of optimization trials
     6. direction: 'maximize' or 'minimize'
    
    Usage example:
        def param_space(trial):
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            }
        
        optimizer = OptunaOptimizer(GradientBoostingRegressor, param_space, scoring='r2', n_trials=100)
        best_model = optimizer.optimize(X_train, y_train)
    """

    def __init__(self, estimator_class, param_space: Callable, scoring: str = "r2",
                 cv: int = 5, n_trials: int = 100, direction: str = "maximize",
                 random_state: int = 42):
        self.estimator_class = estimator_class
        self.param_space = param_space
        self.scoring = scoring
        self.cv = cv
        self.n_trials = n_trials
        self.direction = direction
        self.random_state = random_state
        self.study_ = None
        self.best_model_ = None

    @timer
    def optimize(self, X, y, verbose: bool = True):
        """Run Optuna optimization and return best fitted model."""
        import optuna
        if not verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = self.param_space(trial)
            model = self.estimator_class(**params)
            scores = cross_val_score(model, X, y, cv=self.cv, scoring=self.scoring)
            return scores.mean()

        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        self.study_ = optuna.create_study(direction=self.direction, sampler=sampler)
        self.study_.optimize(objective, n_trials=self.n_trials, show_progress_bar=verbose)

        # Refit best model on full data
        self.best_model_ = self.estimator_class(**self.study_.best_params)
        self.best_model_.fit(X, y)
        logger.info(f"✅ Optuna best score: {self.study_.best_value:.4f} | params: {self.study_.best_params}")
        return self.best_model_

    @property
    def best_params(self) -> Dict[str, Any]:
        if self.study_ is None:
            raise ValueError("Call optimize() first.")
        return self.study_.best_params

    @property
    def best_score(self) -> float:
        if self.study_ is None:
            raise ValueError("Call optimize() first.")
        return self.study_.best_value

    def plot_optimization_history(self):
        """Plotly plot of optimization history."""
        import optuna
        if self.study_ is None:
            raise ValueError("Call optimize() first.")
        return optuna.visualization.plot_optimization_history(self.study_)

    def plot_param_importances(self):
        """Plotly plot of hyperparameter importances."""
        import optuna
        if self.study_ is None:
            raise ValueError("Call optimize() first.")
        return optuna.visualization.plot_param_importances(self.study_)


class HalvingSearchOptimizer:
    """
    Successive Halving for hyperparameter search — faster than full GridSearchCV.
    Starts with many candidates on a small budget, progressively discards the worst.
    
    Dependencies: scikit-learn>=1.1
    
    PARAMETERS:
     1. estimator: sklearn estimator instance
     2. param_grid: dict of parameter grids (same as GridSearchCV)
     3. scoring: sklearn scoring string
     4. cv: number of folds
     5. factor: halving factor (default 3 = discard 2/3 each round)
     6. resource: resource to increase ('n_samples' or estimator param like 'n_estimators')
    
    Usage example:
        optimizer = HalvingSearchOptimizer(
            RandomForestRegressor(),
            {'n_estimators': [50, 100, 200, 500], 'max_depth': [5, 10, 20, None]},
            scoring='r2'
        )
        best_model = optimizer.optimize(X_train, y_train)
    """

    def __init__(self, estimator, param_grid: Dict[str, list], scoring: str = "r2",
                 cv: int = 5, factor: int = 3, resource: str = "n_samples",
                 random_state: int = 42):
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv = cv
        self.factor = factor
        self.resource = resource
        self.random_state = random_state
        self.search_ = None
        self.best_model_ = None

    def optimize(self, X, y, verbose: bool = True):
        """Run halving grid search and return best fitted model."""
        from sklearn.experimental import enable_halving_search_cv  # noqa
        from sklearn.model_selection import HalvingGridSearchCV

        self.search_ = HalvingGridSearchCV(
            self.estimator, self.param_grid, scoring=self.scoring,
            cv=self.cv, factor=self.factor, resource=self.resource,
            random_state=self.random_state, verbose=2 if verbose else 0
        )
        self.search_.fit(X, y)
        self.best_model_ = self.search_.best_estimator_
        logger.info(f"✅ Halving best score: {self.search_.best_score_:.4f} | params: {self.search_.best_params_}")
        return self.best_model_

    @property
    def best_params(self) -> Dict[str, Any]:
        if self.search_ is None:
            raise ValueError("Call optimize() first.")
        return self.search_.best_params_

    @property
    def best_score(self) -> float:
        if self.search_ is None:
            raise ValueError("Call optimize() first.")
        return self.search_.best_score_

    def results_dataframe(self) -> pd.DataFrame:
        """Return full CV results as DataFrame."""
        if self.search_ is None:
            raise ValueError("Call optimize() first.")
        return pd.DataFrame(self.search_.cv_results_)


class EarlyStoppingCV:
    """
    Cross-validation with early stopping — aborts training if no improvement
    for `patience` consecutive iterations. Works with iterative estimators
    (GBM, XGBoost, LightGBM, neural nets).
    
    Dependencies: scikit-learn, numpy, pandas, copy
    
    PARAMETERS:
     1. estimator: estimator with `n_estimators` or similar iterative param
     2. max_iterations: maximum iterations to try
     3. patience: stop if no improvement for this many rounds
     4. scoring: sklearn scoring string
     5. cv: number of folds
    
    Usage example:
        stopper = EarlyStoppingCV(GradientBoostingRegressor(), max_iterations=1000, patience=50)
        best_n, scores = stopper.find_optimal_iterations(X_train, y_train)
    """

    def __init__(self, estimator, max_iterations: int = 1000, patience: int = 50,
                 scoring: str = "r2", cv: int = 5, step: int = 10):
        self.estimator = estimator
        self.max_iterations = max_iterations
        self.patience = patience
        self.scoring = scoring
        self.cv = cv
        self.step = step

    def find_optimal_iterations(self, X, y) -> tuple:
        """
        Incrementally increase n_estimators and evaluate.
        Returns (best_n_estimators, score_history).
        """
        import copy
        best_score = -np.inf
        best_n = self.step
        no_improve = 0
        history = []

        for n in range(self.step, self.max_iterations + 1, self.step):
            est = copy.deepcopy(self.estimator)
            est.set_params(n_estimators=n)
            scores = cross_val_score(est, X, y, cv=self.cv, scoring=self.scoring)
            mean_score = scores.mean()
            history.append({"n_estimators": n, "mean_score": mean_score, "std_score": scores.std()})

            if mean_score > best_score:
                best_score = mean_score
                best_n = n
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= self.patience // self.step:
                logger.info(f"⏹ Early stopping at n_estimators={n}, best={best_n} (score={best_score:.4f})")
                break

        return best_n, pd.DataFrame(history)

    def plot_learning_curve(self, history: pd.DataFrame):
        """Plotly line chart of score vs iterations."""
        import plotly.express as px
        fig = px.line(history, x="n_estimators", y="mean_score",
                      title="Early Stopping Learning Curve")
        fig.update_layout(xaxis_title="n_estimators", yaxis_title=self.scoring)
        return fig


if __name__ == '__main__':
    # Sample data
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.model_selection import train_test_split

    size_df = 500
    X = pd.DataFrame({
        'x1': np.random.randn(size_df),
        'x2': np.random.randn(size_df),
        'x3': np.random.randn(size_df),
    })
    y = 2 * X['x1'] + 0.5 * X['x2'] + np.random.randn(size_df) * 0.3
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Test OptunaOptimizer
    def param_space(trial):
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        }

    optimizer = OptunaOptimizer(GradientBoostingRegressor, param_space, scoring='r2', n_trials=20)
    best_model = optimizer.optimize(X_train, y_train, verbose=False)
    logger.info(f"🎯 Best params: {optimizer.best_params}")

    # Test HalvingSearchOptimizer
    halving = HalvingSearchOptimizer(
        GradientBoostingRegressor(),
        {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 10]},
        scoring='r2'
    )
    halving.optimize(X_train, y_train, verbose=False)

    # Test EarlyStoppingCV
    stopper = EarlyStoppingCV(GradientBoostingRegressor(), max_iterations=200, patience=30, step=20)
    best_n, history = stopper.find_optimal_iterations(X_train, y_train)
    logger.info(f"🎯 Best n_estimators: {best_n}")
