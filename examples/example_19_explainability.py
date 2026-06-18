# -*- coding: utf-8 -*-
"""
Example 19: Explainability — SHAP and LIME
===========================================

Demonstrates model interpretability with:
  1. ShapExplainer — global and local SHAP explanations
  2. LimeExplainer — per-instance LIME explanations

Requirements:
  pip install scomp-link shap lime
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from scomp_link import ShapExplainer, LimeExplainer

# === REGRESSION EXPLAINABILITY ===
print("=" * 60)
print("REGRESSION — SHAP + LIME")
print("=" * 60)

np.random.seed(42)
N = 500
X = pd.DataFrame({
    'income': np.random.randn(N) * 20000 + 50000,
    'age': np.random.randn(N) * 10 + 40,
    'education_years': np.random.randn(N) * 3 + 14,
    'hours_per_week': np.random.randn(N) * 8 + 40,
    'noise': np.random.randn(N),
})
y = 0.8 * X['income'] + 500 * X['age'] + 2000 * X['education_years'] + np.random.randn(N) * 5000

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(n_estimators=100, max_depth=4, random_state=42)
model.fit(X_train, y_train)
print(f"Model R²: {model.score(X_test, y_test):.4f}")

# --- SHAP ---
print("\n🔬 SHAP Analysis")
shap_exp = ShapExplainer(model, X_train[:100])
shap_exp.explain(X_test)
importance = shap_exp.feature_importance()
print(importance.to_string(index=False))

fig = shap_exp.plot_importance(top_n=5)
print(f"✅ SHAP importance plot generated (plotly figure with {len(fig.data)} traces)")

# --- LIME ---
print("\n🔬 LIME Analysis (single instance)")
lime_exp = LimeExplainer(model, X_train, task='regression')
exp = lime_exp.explain_instance(X_test.iloc[0], num_features=5)
print("Instance explanation:")
for feat, weight in exp.as_list():
    direction = "↑" if weight > 0 else "↓"
    print(f"  {direction} {feat}: {weight:.4f}")

fig = lime_exp.plot_explanation(exp)
print(f"✅ LIME explanation plot generated")

# === CLASSIFICATION EXPLAINABILITY ===
print("\n" + "=" * 60)
print("CLASSIFICATION — SHAP + LIME")
print("=" * 60)

X_cls = pd.DataFrame({
    'feature_a': np.random.randn(N),
    'feature_b': np.random.randn(N),
    'feature_c': np.random.randn(N),
})
y_cls = ((X_cls['feature_a'] + 0.5 * X_cls['feature_b']) > 0).astype(int)

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=50, random_state=42)
clf.fit(X_train_c, y_train_c)
print(f"Classifier Accuracy: {clf.score(X_test_c, y_test_c):.4f}")

# SHAP on classifier
shap_cls = ShapExplainer(clf, X_train_c[:50])
shap_cls.explain(X_test_c[:20])
print("\nSHAP importance (classification):")
print(shap_cls.feature_importance().to_string(index=False))

# LIME on classifier
lime_cls = LimeExplainer(clf, X_train_c, task='classification')
exp_cls = lime_cls.explain_instance(X_test_c.iloc[0])
print(f"\nLIME top feature: {exp_cls.as_list()[0]}")
