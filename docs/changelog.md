# Changelog

## 1.1.4 (2026-06-24)

### Added
- Python 3.14 support (experimental — some optional deps not yet available)

### Changed
- Replaced abandoned `pytorch-tabnet` with maintained fork `pytorch-tabnet2` (same API)
- Bumped upper bounds: `torch <2.13`, `transformers <6.0`, `matplotlib <3.11`, `weasyprint <70.0`, `sentence-transformers <6.0`
- TensorFlow excluded from Python 3.14 (no wheels available yet — [tracking issue](https://github.com/tensorflow/tensorflow/issues/102890))
- Updated TabNet import path for pytorch-tabnet2 compatibility

### Fixed
- CI failure on Python 3.14 due to missing TensorFlow wheels

## 1.0.0 (2026-06-18)

### Added
- **Explainability**: `ShapExplainer`, `LimeExplainer` for model interpretability
- **Advanced Tuning**: `OptunaOptimizer`, `HalvingSearchOptimizer`, `EarlyStoppingCV`
- **Drift Detection**: `DriftDetector` with PSI and KS test
- **Pipeline Persistence**: `ScompArtifact` with custom `.scomp` format
- **Feature Engineering**: `FeatureEngineer` (sklearn-compatible)
- **Time Series Forecasting**: `TimeSeriesForecaster` (ARIMA, SARIMA, ETS)
- **Fairness Metrics**: `FairnessMetrics` (demographic parity, disparate impact, equalized odds)
- **Data Quality**: `DataQualityReport` with HTML output
- **Configurable Logging**: `set_verbosity()` to control output (silent/warning/info/debug)
- **Polars backend**: `Preprocessor` uses polars internally for faster processing
- **Documentation**: mkdocs site with full API reference

### Changed
- All `print()` calls replaced with structured logging via `scomp_link.utils.logger`
- `data_processor.py` rewritten with polars (accepts pandas input, returns pandas output)

## 0.1.1 (Initial)

- Core pipeline (`ScompLinkPipeline`)
- Preprocessing (`Preprocessor`)
- Model Factory with decision-tree selection
- Regressor/Classifier Optimizers
- Validation + HTML reports
- Anomaly Detection (tabular + time series)
- Ensemble Learning (voting/stacking)
- Advanced CV (LOOCV, Bootstrap)
- NLP models (contrastive, supervised, unsupervised)
- Image models (CNN, clustering)
