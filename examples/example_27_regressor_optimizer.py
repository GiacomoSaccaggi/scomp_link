# -*- coding: utf-8 -*-
"""
Example 27: RegressorOptimizer — Full Workflow
===============================================

Demonstrates the complete RegressorOptimizer pipeline:
  1. Boruta feature selection (automatic noise removal)
  2. Multi-model grid search (Ridge, Lasso, GradientBoosting)
  3. Optimization time estimation
  4. Model comparison and best model selection
  5. Fit-vs-error visualization

Requirements:
  pip install scomp-link
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor

from scomp_link import RegressorOptimizer
from scomp_link.utils.decorators import timer, memory_usage, validate_args


# --- Helper functions with decorators ---

@memory_usage
def generate_synthetic_data(n_samples: int = 2000) -> pd.DataFrame:
    """Generate a synthetic regression dataset with signal + noise features."""
    np.random.seed(42)

    df = pd.DataFrame({
        # Signal features
        'income': np.random.normal(50000, 15000, n_samples),
        'experience': np.random.uniform(0, 30, n_samples),
        'education_years': np.random.randint(10, 22, n_samples).astype(float),
        'hours_per_week': np.random.normal(40, 8, n_samples),
        # Categorical feature
        'department': np.random.choice(['engineering', 'sales', 'marketing', 'hr'], n_samples),
        # Binary feature
        'is_manager': np.random.choice([True, False], n_samples, p=[0.2, 0.8]),
        # Noise features (should be dropped by Boruta)
        'noise_1': np.random.randn(n_samples),
        'noise_2': np.random.uniform(-1, 1, n_samples),
        'noise_3': np.random.randint(0, 100, n_samples).astype(float),
    })

    # Target: linear combination of signal features + noise
    df['salary'] = (
        0.4 * df['income']
        + 2000 * df['experience']
        + 3000 * df['education_years']
        + 500 * df['hours_per_week']
        + 5000 * df['is_manager'].astype(float)
        + np.random.randn(n_samples) * 8000
    )

    return df


@validate_args(models_dict=lambda x: len(x) >= 2)
def build_models_config(models_dict: dict) -> dict:
    """Validate and return the models configuration."""
    return models_dict


@timer
def run_optimizer(optimizer):
    """Run the full model optimization."""
    optimizer.test_models_regression()
    return optimizer


# --- Main execution ---

if __name__ == '__main__':
    print("=" * 70)
    print("REGRESSOR OPTIMIZER — FULL WORKFLOW")
    print("=" * 70)

    # === 1. Generate data ===
    print("\n--- 1. Generating synthetic data ---")
    df = generate_synthetic_data(2000)
    print(f"  Dataset shape: {df.shape}")
    print(f"  Features: {[c for c in df.columns if c != 'salary']}")
    print(f"  Target: salary (range [{df['salary'].min():.0f}, {df['salary'].max():.0f}])")

    # === 2. Define models to test ===
    print("\n--- 2. Defining models and parameter grids ---")
    models_to_test = build_models_config({
        'Ridge': {
            'model': Ridge(),
            'params_grid': {
                'alpha': [0.1, 1.0, 10.0],
            }
        },
        'Lasso': {
            'model': Lasso(max_iter=5000),
            'params_grid': {
                'alpha': [0.01, 0.1, 1.0],
            }
        },
        'GradientBoosting': {
            'model': GradientBoostingRegressor(random_state=42),
            'params_grid': {
                'n_estimators': [50, 100],
                'max_depth': [3, 5],
                'learning_rate': [0.05, 0.1],
            }
        },
    })
    for name, cfg in models_to_test.items():
        n_combos = 1
        for v in cfg['params_grid'].values():
            n_combos *= len(v)
        print(f"  {name}: {n_combos} parameter combinations")

    # === 3. Create optimizer WITH Boruta feature selection ===
    print("\n--- 3. Initializing RegressorOptimizer (with Boruta) ---")
    x_cols = [c for c in df.columns if c not in ['salary']]

    reg_opt = RegressorOptimizer(
        df=df,
        y_col='salary',
        x_cols=x_cols,
        x_complexity_col='experience',
        models_to_test=models_to_test,
        select_features=True,
    )

    print(f"  Features after Boruta: {reg_opt.X.columns.tolist()}")
    print(f"  Dropped by Boruta: {reg_opt.dropped_columns}")
    print(f"  Numeric cols: {reg_opt.numeric_cols}")
    print(f"  Categorical cols: {reg_opt.categorical_cols}")
    print(f"  Binary cols: {reg_opt.binary_cols}")

    # === 4. Estimate optimization time ===
    print("\n--- 4. Estimating optimization time ---")
    reg_opt.estimate_optimization_time(time_per_combination=2)

    # === 5. Run multi-model optimization ===
    print("\n--- 5. Running multi-model optimization ---")
    reg_opt = run_optimizer(reg_opt)

    # === 6. Results comparison ===
    print("\n--- 6. Results Summary ---")
    print(f"{'Model':<22} {'Best Params':<40} {'CV Score'}")
    print("-" * 80)
    for model_name, result in reg_opt.model_results.items():
        params_str = str(result['Params'])[:38]
        # Extract CV score from the fitted model
        print(f"  {model_name:<20} {params_str:<40}")

    # === 7. Best model identification ===
    print("\n--- 7. Best Model ---")
    best_name = list(reg_opt.model_results.keys())[0]
    best_model = reg_opt.model_results[best_name]['Model']
    print(f"  🏆 Best model: {best_name}")
    print(f"  Model type: {type(best_model).__name__}")

    # === 8. Generate fit-vs-error plot ===
    print("\n--- 8. Generating fit-vs-error visualization ---")
    reg_opt.grafico_fit_con_errore(best_name)
    print(f"  ✅ Plot generated for '{best_name}'")

    # === Summary ===
    print("\n" + "=" * 70)
    print("✅ RegressorOptimizer workflow complete!")
    print(f"   • Boruta removed {len(reg_opt.dropped_columns)} noise features")
    print(f"   • Tested {len(reg_opt.model_results)} models with grid search")
    print(f"   • Best model: {best_name}")
    print("=" * 70)
