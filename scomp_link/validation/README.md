# Validation

Model evaluation, cross-validation, and fairness metrics.

## Modules

| Module | Class | Description |
|--------|-------|-------------|
| `model_validator.py` | `Validator` | Regression/classification metrics, K-Fold CV, HTML report generation |
| `advanced_cv.py` | `AdvancedCV` | LOOCV, Bootstrap with 95% CI |
| `fairness.py` | `FairnessMetrics` | Demographic parity, disparate impact (4/5 rule), equalized odds, equal opportunity |

## Usage

```python
from scomp_link import FairnessMetrics
from scomp_link.validation.advanced_cv import AdvancedCV

# Fairness check
fm = FairnessMetrics(y_true, y_pred, sensitive_feature=df['gender'])
report = fm.compute_all()
print(fm.summary(report))

# Advanced CV
acv = AdvancedCV(model, X, y)
results = acv.evaluate_all()
```
