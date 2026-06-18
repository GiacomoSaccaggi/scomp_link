# Examples

All examples are in the `examples/` directory and can be run standalone.

| # | Example | Description |
|---|---------|-------------|
| 01 | Numerical Small | Econometric model for small datasets |
| 02 | Numerical Medium Lasso | Lasso/ElasticNet feature selection |
| 03 | Numerical Mixed | Mixed feature types with GBM |
| 04 | Classification Small | SVC/KNN/NB pipeline |
| 05 | Classification Large | SGD/GBM for large datasets |
| 06 | Clustering Known | KMeans/Hierarchical |
| 07 | Clustering Unknown | Mean-Shift |
| 08 | Numerical Very Large | SGD Regressor |
| 14 | Ensemble + Advanced CV | Voting/Stacking + LOOCV/Bootstrap |
| 15 | Anomaly Detection | Multi-method consensus (tabular) |
| 16 | TS Anomaly Detection | Autoencoder + moving avg + ARIMA |
| 17 | HTML Reports | Custom Plotly reports |
| 19 | Explainability | SHAP + LIME |
| 20 | Advanced Tuning | Optuna + Halving + Early Stopping |
| 21 | Drift Detection | PSI + KS test |
| 22 | Pipeline Persistence | .scomp format save/load |
| 23 | Feature Engineering | Auto transforms + interactions |
| 24 | TS Forecasting | ARIMA + ETS + walk-forward CV |
| 25 | Fairness Metrics | Bias detection |
| 26 | Data Quality | Profiling + HTML report |

## Running Examples

```bash
python examples/example_01_numerical_small.py
```
