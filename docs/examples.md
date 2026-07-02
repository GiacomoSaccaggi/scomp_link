# Examples

scomp-link includes **34 runnable example scripts** covering all features. Each example is self-contained with synthetic data (no external files required).

```bash
# Run any example
python examples/example_01_numerical_small.py
```

---

## Example Index

### Core Pipeline (01-08)

| # | File | What it demonstrates |
|---|------|---------------------|
| 01 | `example_01_numerical_small.py` | Small regression dataset (<1k rows) |
| 02 | `example_02_numerical_medium_lasso.py` | Medium dataset with Lasso/ElasticNet |
| 03 | `example_03_numerical_mixed_features.py` | Mixed numeric + categorical features |
| 04 | `example_04_classification_small.py` | Small classification (SVC/NB) |
| 05 | `example_05_classification_large.py` | Large classification (GBM/RF) |
| 06 | `example_06_clustering_known.py` | KMeans clustering (known k) |
| 07 | `example_07_clustering_unknown.py` | MeanShift clustering (unknown k) |
| 08 | `example_08_numerical_very_large.py` | Large dataset (>100k) with SGD |

### Text & Image (09-13)

| # | File | What it demonstrates |
|---|------|---------------------|
| 09 | `example_09_text_classification.py` | TF-IDF text classification via pipeline |
| 10 | `example_10_image_classification.py` | CNN image classification via pipeline |
| 11 | `example_11_image_clustering.py` | Image clustering via pipeline |
| 12 | `example_12_text_configuration.py` | Text with contrastive + non-contrastive |
| 13 | `example_13_text_unsupervised.py` | Unsupervised text clustering |

### Advanced ML (14-18)

| # | File | What it demonstrates |
|---|------|---------------------|
| 14 | `example_14_ensemble_advanced_cv.py` | Ensemble (voting/stacking) + AdvancedCV |
| 15 | `example_15_anomaly_detection.py` | All 4 methods: IForest, LOF, TabNet, Transformer |
| 16 | `example_16_ts_anomaly_detection.py` | Time series anomaly (Conv1D + ARIMA) |
| 17 | `example_17_report_html.py` | Full HTML report builder demo |
| 18 | `example_18_contrastive_text_example.py` | BERT contrastive learning |

### Visualization & Utils (19)

| # | File | What it demonstrates |
|---|------|---------------------|
| 19 | `example_19_rawgraphs.py` | All 31 RAWGraphs SVG chart types |
| 19 | `example_19_explainability.py` | SHAP + LIME explanations |

### New Features (20-26)

| # | File | What it demonstrates |
|---|------|---------------------|
| 20 | `example_20_advanced_tuning.py` | Optuna + HalvingSearch tuning |
| 21 | `example_21_drift_detection.py` | DriftDetector (PSI + KS) |
| 22 | `example_22_pipeline_persistence.py` | ScompArtifact save/load/predict |
| 23 | `example_23_feature_engineering.py` | FeatureEngineer (all transforms) |
| 24 | `example_24_time_series_forecasting.py` | TimeSeriesForecaster (ARIMA + ETS) |
| 25 | `example_25_fairness_metrics.py` | FairnessMetrics (demographic parity, DI, equalized odds) |
| 26 | `example_26_data_quality.py` | DataQualityReport HTML generation |

### Coverage Boost (27-34)

| # | File | What it demonstrates |
|---|------|---------------------|
| 27 | `example_27_regressor_optimizer.py` | RegressorOptimizer + Boruta + GridSearch |
| 28 | `example_28_classifier_optimizer.py` | ClassifierOptimizer + multi-model |
| 29 | `example_29_validation_advanced_cv.py` | Validator + AdvancedCV + PDF export |
| 30 | `example_30_core_advanced_pipeline.py` | Pipeline with ensemble + advanced_cv |
| 31 | `example_31_cli_programmatic.py` | All 12 original CLI commands programmatically |
| 32 | `example_32_highcharts_report.py` | Streamgraph + Heatmap + Gantt in report |
| 33 | `example_33_supervised_text.py` | SpacyEmbeddingModel (spaCy v3 textcat) |
| 34 | `example_34_unsupervised_text.py` | TextEmbeddingClustering (sentence-transformers) |

---

## Running All Examples

```bash
# Run all via tox (multi-Python-version)
tox

# Run individually
python examples/example_27_regressor_optimizer.py

# Run with coverage
python -m coverage run --source=scomp_link examples/example_27_regressor_optimizer.py
```
