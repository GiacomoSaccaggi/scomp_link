# -*- coding: utf-8 -*-
"""
Example 25: Fairness & Bias Metrics
====================================

Demonstrates FairnessMetrics on a loan approval scenario:
  1. Demographic Parity
  2. Disparate Impact (4/5 rule)
  3. Equalized Odds
  4. Equal Opportunity

Requirements:
  pip install scomp-link
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scomp_link import FairnessMetrics

np.random.seed(42)
N = 2000

# --- Simulate loan approval data with bias ---
gender = np.random.choice(['male', 'female'], N)
income = np.random.lognormal(10.5, 0.5, N)
credit_score = np.random.normal(700, 50, N)
# True qualification (income + credit)
qualified = ((income > 40000) & (credit_score > 650)).astype(int)

# Train a classifier
X = pd.DataFrame({'income': income, 'credit_score': credit_score, 'is_male': (gender == 'male').astype(int)})
X_train, X_test, y_train, y_test = train_test_split(X, qualified, test_size=0.3, random_state=42)
gender_test = gender[X_test.index]

clf = RandomForestClassifier(n_estimators=50, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("=" * 60)
print("FAIRNESS & BIAS METRICS — Loan Approval")
print("=" * 60)
print(f"Accuracy: {(y_pred == y_test).mean():.3f}")
print(f"Groups: {np.unique(gender_test)}")
print(f"Group sizes: male={sum(gender_test=='male')}, female={sum(gender_test=='female')}")

# --- Compute fairness metrics ---
fm = FairnessMetrics(y_test, y_pred, sensitive_feature=gender_test)
report = fm.compute_all()

# --- Results ---
print("\n--- Demographic Parity ---")
dp = report['demographic_parity']
for group, rate in dp['selection_rates'].items():
    print(f"  {group}: approval rate = {rate:.3f}")
print(f"  Ratio: {dp['dp_ratio']:.3f} {'✅ FAIR' if dp['fair'] else '⚠️ UNFAIR'}")

print("\n--- Disparate Impact (4/5 Rule) ---")
di = report['disparate_impact']
print(f"  DI Ratio: {di['di_ratio']:.3f}")
print(f"  4/5 Rule: {'✅ PASS' if di['four_fifths_rule'] else '⚠️ FAIL'}")

print("\n--- Equalized Odds ---")
eo = report['equalized_odds']
print(f"  TPR per group: {eo['tpr']}")
print(f"  FPR per group: {eo['fpr']}")
print(f"  TPR diff: {eo['tpr_diff']:.3f}, FPR diff: {eo['fpr_diff']:.3f}")
print(f"  {'✅ FAIR' if eo['fair'] else '⚠️ UNFAIR'}")

print("\n--- Summary Table ---")
print(fm.summary(report).to_string(index=False))

# --- Plot ---
fig = fm.plot_fairness_report(report)
print(f"\n✅ Fairness report plot generated")
