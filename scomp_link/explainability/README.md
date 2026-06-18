# Explainability

Model interpretability via SHAP and LIME.

## Modules

| Module | Class | Description |
|--------|-------|-------------|
| `explainer.py` | `ShapExplainer` | Model-agnostic SHAP explanations, feature importance, beeswarm/waterfall plots |
| `explainer.py` | `LimeExplainer` | Per-instance LIME explanations, aggregated importance |

## Usage

```python
from scomp_link import ShapExplainer, LimeExplainer

shap_exp = ShapExplainer(model, X_train[:100])
shap_exp.explain(X_test)
importance = shap_exp.feature_importance()
fig = shap_exp.plot_importance()

lime_exp = LimeExplainer(model, X_train, task='regression')
exp = lime_exp.explain_instance(X_test.iloc[0])
fig = lime_exp.plot_explanation(exp)
```
