# Monitoring

Data drift detection for production model monitoring.

## Modules

| Module | Class | Description |
|--------|-------|-------------|
| `drift_detector.py` | `DriftDetector` | PSI + KS test per feature, drift report, plotly visualizations |

## Usage

```python
from scomp_link import DriftDetector

detector = DriftDetector(X_train, psi_threshold=0.2)
report = detector.detect(X_production)
summary = detector.summary(report)
fig = detector.plot_drift_report(report)
```
