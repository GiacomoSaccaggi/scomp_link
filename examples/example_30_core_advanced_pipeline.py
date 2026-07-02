# -*- coding: utf-8 -*-
"""
Example 30: ScompLinkPipeline — Advanced Features
===================================================

Demonstrates the full ScompLinkPipeline with all advanced flags:
  1. Regression with models_to_test + use_ensemble + advanced_cv
  2. Classification with models_to_test + ensemble (stacking)
  3. Pipeline save_model / load_model flow
  4. Ensemble scores and advanced CV results inspection

Requirements:
  pip install scomp-link
"""

import os
import tempfile
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import (
    GradientBoostingRegressor, GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.neighbors import KNeighborsClassifier

from scomp_link import ScompLinkPipeline
from scomp_link.utils.decorators import timer, log_call, memory_usage


# --- Helper functions with decorators ---

@memory_usage
def generate_regression_dataset(n: int = 800) -> pd.DataFrame:
    """Generate a regression dataset for pipeline testing."""
    np.random.seed(42)
    df = pd.DataFrame({
        'temperature': np.random.normal(20, 8, n),
        'humidity': np.random.normal(60, 15, n),
        'pressure': np.random.normal(1013, 10, n),
        'wind_speed': np.random.exponential(5, n),
        'elevation': np.random.uniform(0, 3000, n),
    })
    df['energy_output'] = (
        2.5 * df['temperature']
        - 0.8 * df['humidity']
        + 0.1 * df['pressure']
        + 1.5 * df['wind_speed']
        + 0.01 * df['elevation']
        + np.random.randn(n) * 5
    )
    return df


@memory_usage
def generate_classification_dataset(n: int = 600) -> pd.DataFrame:
    """Generate a binary classification dataset for pipeline testing."""
    np.random.seed(123)
    df = pd.DataFrame({
        'score_a': np.random.randn(n),
        'score_b': np.random.randn(n),
        'score_c': np.random.randn(n) * 0.5,
        'metric_x': np.random.uniform(0, 10, n),
    })
    df['outcome'] = ((df['score_a'] + 0.7 * df['score_b'] + 0.3 * df['metric_x']) > 2).astype(int)
    return df


@timer
@log_call
def run_regression_pipeline(df, models_to_test):
    """Run the full regression pipeline with ensemble + advanced CV."""
    pipe = ScompLinkPipeline("Energy Output Prediction")
    pipe.set_objectives(["Minimize RMSE", "Maximize R²"])
    pipe.import_and_clean_data(df)
    pipe.select_variables(target_col='energy_output')
    pipe.choose_model("numerical_prediction")

    results = pipe.run_pipeline(
        task_type="regression",
        models_to_test=models_to_test,
        use_ensemble=True,
        ensemble_strategy='voting',
        test_size=0.25,
    )
    return pipe, results


@timer
@log_call
def run_classification_pipeline(df, models_to_test):
    """Run the full classification pipeline with stacking ensemble + advanced CV."""
    pipe = ScompLinkPipeline("Outcome Classification")
    pipe.set_objectives(["Maximize F1"])
    pipe.import_and_clean_data(df)
    pipe.select_variables(target_col='outcome')
    pipe.choose_model("categorical_known")

    results = pipe.run_pipeline(
        task_type="classification",
        models_to_test=models_to_test,
        use_ensemble=True,
        ensemble_strategy='stacking',
        test_size=0.25,
    )
    return pipe, results


@timer
def run_simple_with_advanced_cv(df):
    """Run a simple pipeline (no models_to_test) with advanced_cv enabled."""
    pipe = ScompLinkPipeline("Simple + Advanced CV")
    pipe.set_objectives(["Minimize RMSE"])
    pipe.import_and_clean_data(df)
    pipe.select_variables(target_col='energy_output')
    pipe.choose_model("numerical_prediction")

    results = pipe.run_pipeline(
        task_type="regression",
        advanced_cv=True,
        cv_methods=['bootstrap'],
        bootstrap_iterations=100,
        test_size=0.25,
    )
    return pipe, results


# --- Main execution ---

if __name__ == '__main__':
    print("=" * 70)
    print("SCOMP-LINK PIPELINE — ADVANCED FEATURES")
    print("=" * 70)

    # ═══════════════════════════════════════════════════════
    # SCENARIO 1: REGRESSION + ENSEMBLE + MODELS_TO_TEST
    # ═══════════════════════════════════════════════════════

    print("\n" + "═" * 50)
    print("SCENARIO 1: REGRESSION + ENSEMBLE")
    print("═" * 50)

    df_reg = generate_regression_dataset(800)
    print(f"\n  Dataset: {df_reg.shape}")

    reg_models = {
        'Ridge': {
            'model': Ridge(),
            'params_grid': {'alpha': [0.1, 1.0, 10.0]},
        },
        'Lasso': {
            'model': Lasso(max_iter=5000),
            'params_grid': {'alpha': [0.01, 0.1, 1.0]},
        },
        'GBR': {
            'model': GradientBoostingRegressor(random_state=42),
            'params_grid': {
                'n_estimators': [50, 100],
                'max_depth': [3, 5],
            },
        },
    }

    pipe_reg, results_reg = run_regression_pipeline(df_reg, reg_models)

    print("\n  --- Regression Results ---")
    if 'optimizer_results' in results_reg:
        print(f"  Models tested: {list(results_reg['optimizer_results'].keys())}")
    if 'ensemble_scores' in results_reg:
        ens = results_reg['ensemble_scores']
        print(f"  🎯 Ensemble CV Score: {ens['mean_score']:.4f} (±{ens['std_score']:.4f})")

    # ═══════════════════════════════════════════════════════
    # SCENARIO 2: CLASSIFICATION + STACKING ENSEMBLE
    # ═══════════════════════════════════════════════════════

    print("\n\n" + "═" * 50)
    print("SCENARIO 2: CLASSIFICATION + STACKING ENSEMBLE")
    print("═" * 50)

    df_cls = generate_classification_dataset(600)
    print(f"\n  Dataset: {df_cls.shape}")
    print(f"  Class distribution: {df_cls['outcome'].value_counts().to_dict()}")

    cls_models = {
        'RandomForest': {
            'model': RandomForestClassifier(random_state=42),
            'params_grid': {
                'n_estimators': [30, 60],
                'max_depth': [5, 10],
            },
        },
        'GradientBoosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params_grid': {
                'n_estimators': [30, 60],
                'learning_rate': [0.05, 0.1],
            },
        },
        'KNeighbors': {
            'model': KNeighborsClassifier(),
            'params_grid': {
                'n_neighbors': [3, 5, 7],
            },
        },
    }

    pipe_cls, results_cls = run_classification_pipeline(df_cls, cls_models)

    print("\n  --- Classification Results ---")
    if 'optimizer_results' in results_cls:
        print(f"  Models tested: {list(results_cls['optimizer_results'].keys())}")
    if 'ensemble_scores' in results_cls:
        ens_cls = results_cls['ensemble_scores']
        print(f"  🎯 Stacking Ensemble CV Score: {ens_cls['mean_score']:.4f} (±{ens_cls['std_score']:.4f})")

    # ═══════════════════════════════════════════════════════
    # SCENARIO 3: SIMPLE PIPELINE + ADVANCED CV
    # ═══════════════════════════════════════════════════════

    print("\n\n" + "═" * 50)
    print("SCENARIO 3: SIMPLE PIPELINE + ADVANCED CV")
    print("═" * 50)

    pipe_adv, results_adv = run_simple_with_advanced_cv(df_reg)

    print("\n  --- Advanced CV Results ---")
    print(f"  Status: {results_adv['status']}")
    print(f"  Model type: {results_adv['model_type']}")
    print(f"  Metrics: {results_adv['metrics']}")
    if results_adv.get('advanced_cv'):
        for key, cv_result in results_adv['advanced_cv'].items():
            print(f"  {cv_result['method']}: {cv_result['mean_score']:.4f} (±{cv_result['std_score']:.4f})")

    # ═══════════════════════════════════════════════════════
    # SCENARIO 4: SAVE / LOAD MODEL
    # ═══════════════════════════════════════════════════════

    print("\n\n" + "═" * 50)
    print("SCENARIO 4: SAVE / LOAD MODEL")
    print("═" * 50)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Save
        save_path = pipe_adv.save_model(tmpdir)
        files = os.listdir(tmpdir)
        print(f"\n  Saved to: {tmpdir}")
        print(f"  Files: {files}")

        # Load
        pipe_loaded = ScompLinkPipeline("Loaded Pipeline")
        pipe_loaded.load_model(tmpdir)
        print(f"  ✅ Model loaded successfully: {type(pipe_loaded.model).__name__}")

    # === Summary ===
    print("\n" + "=" * 70)
    print("✅ ScompLinkPipeline advanced features demo complete!")
    print("   • Regression: models_to_test + voting ensemble")
    print("   • Classification: models_to_test + stacking ensemble")
    print("   • Simple pipeline with advanced_cv (bootstrap)")
    print("   • Model save/load round-trip")
    print("=" * 70)

    # Cleanup generated report file
    import glob
    for f in glob.glob("ScompLink_Validation_Report*.html"):
        os.unlink(f)
