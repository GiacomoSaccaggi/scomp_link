# -*- coding: utf-8 -*-
"""
Example 28: ClassifierOptimizer — Full Workflow
================================================

Demonstrates the complete ClassifierOptimizer pipeline:
  1. Synthetic multiclass dataset (4 classes, mixed feature types)
  2. Boruta-based feature selection for classification
  3. Multi-model grid search (RandomForest, GradientBoosting, KNeighbors)
  4. Classification report and confusion matrix per model
  5. Results comparison

Requirements:
  pip install scomp-link
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier

from scomp_link import ClassifierOptimizer
from scomp_link.utils.decorators import timer, memory_usage, validate_args


# --- Helper functions with decorators ---

@memory_usage
def generate_multiclass_data(n_samples: int = 1500) -> pd.DataFrame:
    """Generate a synthetic 4-class classification dataset."""
    np.random.seed(42)

    # Create clusters for 4 classes
    n_per_class = n_samples // 4
    classes = []
    data_rows = []

    for cls_id, (cx, cy) in enumerate([(0, 0), (4, 4), (-4, 4), (0, -4)]):
        for _ in range(n_per_class):
            data_rows.append({
                'feature_x': np.random.normal(cx, 1.2),
                'feature_y': np.random.normal(cy, 1.2),
                'magnitude': np.random.exponential(2.0) + cls_id * 0.5,
                'category': np.random.choice(['alpha', 'beta', 'gamma', 'delta']),
                'is_premium': np.random.random() > (0.5 + cls_id * 0.1),
                # Noise features
                'noise_a': np.random.randn(),
                'noise_b': np.random.uniform(-5, 5),
            })
            classes.append(f'class_{cls_id}')

    df = pd.DataFrame(data_rows)
    df['target'] = classes
    return df


@validate_args(n_models=lambda x: x >= 2)
def validate_model_count(n_models: int) -> bool:
    """Ensure we have enough models for meaningful comparison."""
    return True


@timer
def run_classification(optimizer):
    """Run the full classification optimization."""
    optimizer.test_models_classification()
    return optimizer


# --- Main execution ---

if __name__ == '__main__':
    print("=" * 70)
    print("CLASSIFIER OPTIMIZER — FULL WORKFLOW")
    print("=" * 70)

    # === 1. Generate data ===
    print("\n--- 1. Generating multiclass dataset ---")
    df = generate_multiclass_data(1500)
    print(f"  Dataset shape: {df.shape}")
    print(f"  Classes: {df['target'].unique().tolist()}")
    print(f"  Class distribution:")
    for cls, count in df['target'].value_counts().items():
        print(f"    {cls}: {count} samples")

    # === 2. Define models to test ===
    print("\n--- 2. Defining classifier models and grids ---")
    models_to_test = {
        'RandomForest': {
            'model': RandomForestClassifier(random_state=42),
            'params_grid': {
                'n_estimators': [50, 100],
                'max_depth': [5, 10],
                'min_samples_split': [2, 5],
            }
        },
        'GradientBoosting': {
            'model': GradientBoostingClassifier(random_state=42),
            'params_grid': {
                'n_estimators': [50, 100],
                'max_depth': [3, 5],
                'learning_rate': [0.05, 0.1],
            }
        },
        'KNeighbors': {
            'model': KNeighborsClassifier(),
            'params_grid': {
                'n_neighbors': [3, 5, 7],
                'weights': ['uniform', 'distance'],
            }
        },
    }

    validate_model_count(len(models_to_test))

    for name, cfg in models_to_test.items():
        n_combos = 1
        for v in cfg['params_grid'].values():
            n_combos *= len(v)
        print(f"  {name}: {n_combos} parameter combinations")

    # === 3. Create optimizer with feature selection ===
    print("\n--- 3. Initializing ClassifierOptimizer (with Boruta) ---")
    x_cols = [c for c in df.columns if c != 'target']

    clf_opt = ClassifierOptimizer(
        df=df,
        y_col='target',
        x_cols=x_cols,
        models_to_test=models_to_test,
        select_features=True,
    )

    print(f"  Selected features: {clf_opt.X.columns.tolist()}")
    print(f"  Train size: {len(clf_opt.X_train)}, Test size: {len(clf_opt.X_test)}")
    print(f"  Numeric cols: {clf_opt.numeric_cols}")
    print(f"  Categorical cols: {clf_opt.categorical_cols}")

    # === 4. Run multi-model optimization ===
    print("\n--- 4. Running multi-model classification ---")
    clf_opt = run_classification(clf_opt)

    # === 5. Results per model ===
    print("\n--- 5. Results per Model ---")
    for model_name, result in clf_opt.model_results.items():
        print(f"\n  {'='*50}")
        print(f"  🔬 {model_name}")
        print(f"  {'='*50}")
        print(f"  Best Params: {result['Params']}")
        print(f"\n  Classification Report:")
        print(f"  {result['Report']}")
        print(f"\n  Confusion Matrix:")
        print(f"  {result['Confusion_Matrix']}")

    # === 6. Print formatted summary ===
    print("\n--- 6. Summary Comparison ---")
    clf_opt.print_results()

    # === 7. Accuracy comparison ===
    print("\n--- 7. Accuracy Ranking ---")
    from sklearn.metrics import accuracy_score
    accuracies = {}
    for model_name, result in clf_opt.model_results.items():
        acc = accuracy_score(result['True_Test'], result['Fitted_Test'])
        accuracies[model_name] = acc

    for rank, (name, acc) in enumerate(sorted(accuracies.items(), key=lambda x: -x[1]), 1):
        medal = "🥇" if rank == 1 else "🥈" if rank == 2 else "🥉"
        print(f"  {medal} #{rank} {name}: {acc:.4f}")

    # === Summary ===
    print("\n" + "=" * 70)
    best_name = max(accuracies, key=accuracies.get)
    print(f"✅ ClassifierOptimizer workflow complete!")
    print(f"   • Tested {len(clf_opt.model_results)} models with grid search")
    print(f"   • Best model: {best_name} (accuracy={accuracies[best_name]:.4f})")
    print(f"   • All models produced classification reports + confusion matrices")
    print("=" * 70)
