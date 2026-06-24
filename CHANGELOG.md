# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.4] - 2026-06-24

### Added
- Python 3.14 support (experimental — some optional deps not yet available)

### Changed
- Replaced abandoned `pytorch-tabnet` with maintained fork `pytorch-tabnet2` (same API)
- Bumped upper bounds: `torch <2.13`, `transformers <6.0`, `matplotlib <3.11`, `weasyprint <70.0`, `sentence-transformers <6.0`
- TensorFlow excluded from Python 3.14 (no wheels available yet — [tracking issue](https://github.com/tensorflow/tensorflow/issues/102890))
- `tf-keras` restricted to Python <3.14
- Updated TabNet import path for pytorch-tabnet2 compatibility

### Fixed
- CI failure on Python 3.14 due to missing TensorFlow wheels
- Improved error message for TensorFlow ImportError with Python 3.14 guidance

## [1.0.0] - 2026-06-18

### Added
- **CLI**: 11 commands (`run`, `predict`, `explain`, `engineer`, `forecast`, `anomaly`, `drift`, `fairness`, `quality`, `compare`, `info`)
- **Explainability**: `ShapExplainer`, `LimeExplainer` for model interpretability
- **Advanced Tuning**: `OptunaOptimizer`, `HalvingSearchOptimizer`, `EarlyStoppingCV`
- **Drift Detection**: `DriftDetector` with PSI and KS test
- **Pipeline Persistence**: `ScompArtifact` with custom `.scomp` format
- **Feature Engineering**: `FeatureEngineer` (sklearn-compatible fit/transform)
- **Time Series Forecasting**: `TimeSeriesForecaster` (ARIMA, SARIMA, ETS, walk-forward CV)
- **Fairness Metrics**: `FairnessMetrics` (demographic parity, disparate impact, equalized odds)
- **Data Quality**: `DataQualityReport` with standalone HTML output
- **Configurable Logging**: `set_verbosity()` to control output (silent/warning/info/debug)
- **Polars backend**: `Preprocessor` uses polars internally for faster processing
- **Documentation**: mkdocs site with full API and CLI reference
- **CI/CD**: GitHub Actions with matrix testing (py310-313), Codecov, auto-publish to PyPI

### Changed
- All `print()` calls replaced with structured logging via `scomp_link.utils.logger`
- `data_processor.py` rewritten with polars (accepts pandas input, returns pandas output)
- All docstrings translated to English

## [0.1.1] - 2026-02-26

### Added
- Core pipeline (`ScompLinkPipeline`)
- Preprocessing (`Preprocessor`)
- Model Factory with decision-tree-based selection
- `RegressorOptimizer` / `ClassifierOptimizer` with GridSearchCV
- Validation + HTML reports (`Validator`, `ScompLinkHTMLReport`)
- Anomaly Detection: tabular (`AnomalyDetector`) + time series (`TimeSeriesAnomalyDetector`)
- Ensemble Learning: voting and stacking (`EnsembleOptimizer`)
- Advanced CV: LOOCV, Bootstrap (`AdvancedCV`)
- NLP models: contrastive, supervised, unsupervised text
- Image models: CNN classification, clustering
- 18 standalone examples
