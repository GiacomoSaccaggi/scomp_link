# -*- coding: utf-8 -*-
"""
Example 20: Advanced Hyperparameter Tuning
==========================================

Demonstrates 3 strategies beyond GridSearchCV:
  1. OptunaOptimizer — Bayesian optimization (TPE sampler)
  2. HalvingSearchOptimizer — Successive Halving (fast elimination)
  3. EarlyStoppingCV — Patience-based iteration search

Requirements:
  pip install scomp-link optuna
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from scomp_link.models.advanced_tuning import OptunaOptimizer, HalvingSearchOptimizer, EarlyStoppingCV

# --- Generate synthetic data ---
np.random.seed(42)
N = 1000
X = pd.DataFrame({
    'x1': np.random.randn(N),
    'x2': np.random.randn(N),
    'x3': np.random.randn(N),
    'x4': np.random.randn(N),
    'x5': np.random.randn(N),
})
y = 3 * X['x1'] + 1.5 * X['x2'] - 0.5 * X['x3'] + np.random.randn(N) * 0.5

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 1. OPTUNA (Bayesian Optimization) ===
print("=" * 60)
print("1. OptunaOptimizer — Bayesian HPO")
print("=" * 60)


def param_space(trial):
    return {
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'max_depth': trial.suggest_int('max_depth', 2, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
    }


optimizer = OptunaOptimizer(
    GradientBoostingRegressor, param_space,
    scoring='r2', n_trials=30, cv=5
)
best_model = optimizer.optimize(X_train, y_train, verbose=False)
test_score = best_model.score(X_test, y_test)
print(f"🎯 Test R²: {test_score:.4f}")
print(f"   Best params: {optimizer.best_params}")

# === 2. HALVING GRID SEARCH ===
print("\n" + "=" * 60)
print("2. HalvingSearchOptimizer — Successive Halving")
print("=" * 60)

halving = HalvingSearchOptimizer(
    GradientBoostingRegressor(random_state=42),
    param_grid={
        'n_estimators': [50, 100, 150, 200, 300],
        'max_depth': [2, 3, 4, 5, 6],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
    },
    scoring='r2', cv=3, factor=3
)
best_halving = halving.optimize(X_train, y_train, verbose=False)
test_score_h = best_halving.score(X_test, y_test)
print(f"🎯 Test R²: {test_score_h:.4f}")

results_df = halving.results_dataframe()
print(f"   Candidates evaluated: {len(results_df)}")

# === 3. EARLY STOPPING CV ===
print("\n" + "=" * 60)
print("3. EarlyStoppingCV — Patience-based")
print("=" * 60)

stopper = EarlyStoppingCV(
    GradientBoostingRegressor(learning_rate=0.05, max_depth=4, random_state=42),
    max_iterations=500, patience=50, step=20, cv=5, scoring='r2'
)
best_n, history = stopper.find_optimal_iterations(X_train, y_train)
print(f"🎯 Optimal n_estimators: {best_n}")
print(f"   Score at optimum: {history[history['n_estimators'] == best_n]['mean_score'].values[0]:.4f}")

fig = stopper.plot_learning_curve(history)
print(f"✅ Learning curve plot generated ({len(history)} points)")

# === COMPARISON ===
print("\n" + "=" * 60)
print("COMPARISON")
print("=" * 60)
print(f"  Optuna R²:       {test_score:.4f}")
print(f"  Halving R²:      {test_score_h:.4f}")
final_model = GradientBoostingRegressor(n_estimators=best_n, learning_rate=0.05, max_depth=4, random_state=42)
final_model.fit(X_train, y_train)
print(f"  EarlyStopping R²: {final_model.score(X_test, y_test):.4f}")
