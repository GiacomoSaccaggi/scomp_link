# Changelog

All notable changes to scomp-link are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)

## [1.2.8] - 2026-07-07

### Fixed
- CI coverage now includes all examples (72% reported to Codecov)
- Fixed `example_11_image_clustering.py` silhouette_score crash
- Fixed `mcp_server.py` FastMCP constructor for mcp>=1.28 (`description` → `instructions`)
- Skip BERT examples (09, 12, 18) in CI to prevent 6h timeout

### Added
- MCP server test suite: 25 tests covering all 14 tools
- `pre-commit` added to dev dependencies

### Changed
- CI installs `[dev,mcp]` extras for full test coverage
- `bump_version.py` now covers all 9 version locations
- All version strings aligned to single source of truth

## [1.2.7] - 2026-07-07

### Fixed
- Cursor plugin version alignment
- HF Space app version alignment

## [1.2.6] - 2026-07-06

### Added
- Demo page with asciinema terminal recording
- GitHub Pages deployment in CI

### Changed
- Demo report and terminal embedded in `docs/demo/`

## [1.2.5] - 2026-07-02

### Added
- `.well-known/mcp.json` and `server.json` for MCP discovery
- `ai-catalog.json` for AI agent discovery
- Cursor marketplace plugin (`.cursor-plugin/plugin.json`)
- Docker multi-registry publishing (GHCR + DockerHub)
- Smithery auto-publish in CI
- HuggingFace Skill upload in CI
- `docs/comparison.md` — comparison with other ML frameworks

### Fixed
- CLI `compare` command output format
- Skill YAML config examples

## [1.2.4] - 2026-07-02

### Fixed
- Pillow version range simplified (>=9.0/<13.0 for all Python versions)
- uv.lock sync with pyproject.toml

## [1.2.3] - 2026-07-02

### Added
- `llms.txt` discovery file
- `AGENTS.md` for AI coding agent instructions
- GitHub Issue templates (bug report, feature request)
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`
- `dependabot.yml` for automated dependency updates
- CodeQL security scanning in CI

## [1.2.2] - 2026-07-02

### Fixed
- SKILL.md links use absolute GitHub URLs (fixes Smithery navigation)
- HuggingFace namespace corrected in CI workflow

### Changed
- Version centralized via `scripts/bump_version.py`

## [1.2.1] - 2026-07-01

### Added
- **MCP Server** with 15 tools, 3 resources, 4 prompts (`scomp-link mcp`)
- **11 new CLI commands**: text, cluster, tune, validate, monitor, serve, export, pipeline, describe, list-models, check-deps
- `--format json|csv|table` on run/validate/tune/describe
- `--plot` flag on forecast/drift/anomaly/cluster/compare
- YAML-driven pipelines (`scomp-link pipeline --config`)
- REST API serving (`scomp-link serve`)
- Model export to pickle/joblib/ONNX (`scomp-link export`)
- Agent Skill (SKILL.md) with decision tree, CLI reference, visualization guide, workflow patterns
- Discovery files: `llms.txt`, `AGENTS.md`, `ai-catalog.json`, `server.json`, `smithery.yaml`
- GitHub Actions CI for automatic publishing (PyPI + MCP Registry + HuggingFace)
- 8 new examples (27-34)
- 29 new tests for CLI commands + choose_model branches
- Pre-commit hooks with Ruff linting

### Fixed
- `supervised_text.py` migrated from spaCy v2 to v3 API
- `core.py` handles DataFrames with image/array columns
- `pdf_converter.py` catches OSError from weasyprint
- `_select_language` refactored: 60+ elif → dynamic importlib (6 lines)

### Changed
- CLI expanded from 13 to 25 commands
- Coverage: 65% → 81% (279 tests + 34 examples)
- Dependencies bumped: pillow <13, scikit-learn <1.8, pytorch-tabnet >=4.1, numpy <2.3, pandas <2.4
- Added optional extras: `[mcp]`, `[serve]`

## [1.2.0] - 2026-06-24

### Added
- RAWGraphs SVG chart library (31 chart types)
- DataQualityReport HTML generation
- FeatureEngineer (sklearn-compatible)
- TimeSeriesForecaster (ARIMA, Exp. Smoothing, auto)
- FairnessMetrics (demographic parity, disparate impact, equalized odds)
- DriftDetector (PSI + KS test)
- ShapExplainer and LimeExplainer
- ScompArtifact (.scomp persistence format)
- OptunaOptimizer and HalvingSearchOptimizer
- Polars backend for Preprocessor

## [1.1.0] - 2026-05-06

### Added
- CLI with 13 commands
- EnsembleOptimizer (voting/stacking)
- AdvancedCV (LOOCV, Bootstrap)
- AnomalyDetector (IForest, LOF, TabNet, Transformer)
- TimeSeriesAnomalyDetector
- Highcharts visualizations (streamgraph, heatmap, gantt)

## [1.0.0] - 2026-02-27

### Added
- Initial release
- ScompLinkPipeline orchestrator
- RegressorOptimizer with Boruta feature selection
- ClassifierOptimizer
- ModelFactory decision tree
- ScompLinkHTMLReport builder
- Plotly utils (histogram, barchart, linechart, area_chart)
- 18 example scripts
