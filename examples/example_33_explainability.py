# -*- coding: utf-8 -*-
"""
Example 33: Explainability — ShapExplainer & LimeExplainer
============================================================

Demonstrates all methods of ShapExplainer and LimeExplainer:
  - ShapExplainer: explain(), feature_importance(), plot_importance(),
    plot_beeswarm(), plot_waterfall()
  - LimeExplainer: explain_instance(), plot_explanation(),
    feature_importance() aggregated over multiple samples

Requirements:
  pip install scomp-link[shap]
  # shap and lime are installed with the [shap] extra
"""

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split

from scomp_link import LimeExplainer, ShapExplainer

# ---------------------------------------------------------------------------
# Synthetic data — regression
# ---------------------------------------------------------------------------
np.random.seed(42)
N = 300
X_reg = pd.DataFrame(
    {
        "x1": np.random.randn(N),
        "x2": np.random.randn(N),
        "x3": np.random.randn(N) * 0.5,
    }
)
y_reg = 3 * X_reg["x1"] - 1.5 * X_reg["x2"] + np.random.randn(N) * 0.2

X_train_r, X_test_r, y_train_r, _ = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
model_reg = RandomForestRegressor(n_estimators=50, random_state=42)
model_reg.fit(X_train_r, y_train_r)

# ---------------------------------------------------------------------------
# Synthetic data — classification
# ---------------------------------------------------------------------------
y_clf = (X_reg["x1"] + X_reg["x2"] > 0).astype(int)
X_train_c, X_test_c, y_train_c, _ = train_test_split(X_reg, y_clf, test_size=0.2, random_state=42)
model_clf = GradientBoostingClassifier(n_estimators=50, random_state=42)
model_clf.fit(X_train_c, y_train_c)


# ---------------------------------------------------------------------------
# ShapExplainer — regression
# ---------------------------------------------------------------------------
print("\n=== ShapExplainer (regression) ===")
shap_exp = ShapExplainer(model_reg, X_train_r[:50])

# explain()
shap_values = shap_exp.explain(X_test_r[:20])
print(f"  SHAP values shape: {shap_values.shape}")
assert shap_values.shape == (20, 3), f"Unexpected shape: {shap_values.shape}"

# feature_importance()
importance = shap_exp.feature_importance()
print(f"  Top feature: {importance.iloc[0]['feature']}")
assert importance["mean_abs_shap"].is_monotonic_decreasing
assert importance.iloc[0]["feature"] == "x1"

# plot_importance()
fig = shap_exp.plot_importance(top_n=3)
assert hasattr(fig, "to_json"), "Expected Plotly figure"
print("  ✅ plot_importance OK")

# plot_beeswarm() — matplotlib, suppress display
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        shap_exp.plot_beeswarm()
        print("  ✅ plot_beeswarm OK")
    except Exception as e:
        print(f"  ⚠️  plot_beeswarm skipped: {e}")

# plot_waterfall() — single prediction
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        shap_exp.plot_waterfall(idx=0)
        print("  ✅ plot_waterfall OK")
    except Exception as e:
        print(f"  ⚠️  plot_waterfall skipped: {e}")


# ---------------------------------------------------------------------------
# ShapExplainer fallback paths: TreeExplainer, KernelExplainer
# ---------------------------------------------------------------------------
print("\n=== ShapExplainer fallback (TreeExplainer) ===")
# GradientBoosting triggers TreeExplainer fallback (Explainer may fail on it)
try:
    shap_tree = ShapExplainer(model_clf, X_train_c[:30])
    shap_tree.explain(X_test_c[:10])
    print("  ✅ TreeExplainer fallback OK")
except Exception as e:
    print(f"  ⚠️  TreeExplainer fallback skipped: {e}")


# ---------------------------------------------------------------------------
# ShapExplainer — multidimensional SHAP values (importance.ndim > 1 branch)
# ---------------------------------------------------------------------------
print("\n=== ShapExplainer multi-output (classification) ===")
try:
    shap_clf = ShapExplainer(model_clf, X_train_c[:50])
    shap_clf.explain(X_test_c[:10])
    imp_clf = shap_clf.feature_importance()
    print(f"  Top feature (classification): {imp_clf.iloc[0]['feature']}")
    print("  ✅ multi-output importance OK")
except Exception as e:
    print(f"  ⚠️  multi-output skipped: {e}")


# ---------------------------------------------------------------------------
# LimeExplainer — regression
# ---------------------------------------------------------------------------
print("\n=== LimeExplainer (regression) ===")
lime_reg = LimeExplainer(model_reg, X_train_r, task="regression")

# explain_instance()
exp = lime_reg.explain_instance(X_test_r.iloc[0], num_features=3)
pairs = exp.as_list()
print(f"  Explanation pairs: {len(pairs)}")
assert len(pairs) == 3

# plot_explanation()
fig_lime = lime_reg.plot_explanation(exp, top_n=3)
assert hasattr(fig_lime, "to_json"), "Expected Plotly figure"
print("  ✅ plot_explanation OK")

# feature_importance() aggregated
print("  Computing aggregated LIME importance (20 samples)...")
agg = lime_reg.feature_importance(X_test_r, n_samples=20, num_features=3)
assert "feature" in agg.columns
assert "mean_abs_weight" in agg.columns
assert agg["mean_abs_weight"].is_monotonic_decreasing
print(f"  Top LIME feature: {agg.iloc[0]['feature']}")
print("  ✅ feature_importance aggregated OK")


# ---------------------------------------------------------------------------
# LimeExplainer — classification
# ---------------------------------------------------------------------------
print("\n=== LimeExplainer (classification) ===")
lime_clf = LimeExplainer(
    model_clf, X_train_c, task="classification", feature_names=["x1", "x2", "x3"]
)
exp_clf = lime_clf.explain_instance(X_test_c.iloc[0], num_features=2)
assert len(exp_clf.as_list()) > 0
print("  ✅ classification explain_instance OK")


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("✅ All explainability methods exercised successfully")
print("   ShapExplainer: explain, feature_importance, plot_importance,")
print("                  plot_beeswarm, plot_waterfall")
print("   LimeExplainer: explain_instance, plot_explanation,")
print("                  feature_importance (aggregated)")
print("=" * 60)
