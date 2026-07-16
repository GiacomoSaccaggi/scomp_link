# Changelog

## 1.3.0 (2026-07-16)

### Added
- **Report Builder MCP tools**: `report_create`, `report_add_section`, `report_add_text`, `report_add_table`, `report_add_chart`, `report_save` — build custom branded HTML reports step-by-step via MCP
- **Configuration system**: `scomp-link init-config` CLI command + persistent config files (`~/.scomp-link/config.yaml` global, `.scomp-link.yaml` local) for corporate branding defaults
- **Footer parametrization**: `footer_html` parameter in `ScompLinkHTMLReport` for custom report footers
- `report_add_chart` supports all 39 chart types across 3 engines (plotly, rawgraphs, highcharts)
- `build_custom_report` MCP prompt for guided report building
- Quick Setup Prompt for AI Agents in README

### Changed
- MCP server now exposes 22 tools (was 16)
- CLI now has 26 commands (was 25)
- `generate_report` MCP tool now accepts `footer_html` parameter

## 1.2.8 (2026-07-07)

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

## 1.2.7 (2026-07-07)

### Fixed
- Cursor plugin version alignment
- HF Space app version alignment

## 1.2.6 (2026-07-06)

### Added
- Demo page with asciinema terminal recording
- GitHub Pages deployment in CI

## 1.2.5 (2026-07-02)

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

## 1.2.4 (2026-07-02)

### Fixed
- Pillow version range simplified
- uv.lock sync with pyproject.toml

## 1.2.3 (2026-07-02)

### Added
- `llms.txt` discovery file
- `AGENTS.md` for AI coding agent instructions
- GitHub Issue templates (bug report, feature request)
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`
- `dependabot.yml` for automated dependency updates
- CodeQL security scanning in CI

## 1.2.2 (2026-07-02)

### Fixed
- SKILL.md links use absolute GitHub URLs (fixes Smithery navigation)
- HuggingFace namespace corrected in CI workflow

### Changed
- Version centralized via `scripts/bump_version.py`

## 1.2.1 (2026-07-01)

### Added
- **MCP Server** with 15 tools, 3 resources, 4 prompts (`scomp-link mcp`)
- **11 new CLI commands**: text, cluster, tune, validate, monitor, serve, export, pipeline, describe, list-models, check-deps
- `--format json|csv|table` on run/validate/tune/describe
- `--plot` flag on forecast/drift/anomaly/cluster/compare
- YAML-driven pipelines (`scomp-link pipeline --config`)
- REST API serving (`scomp-link serve`)
- Model export to pickle/joblib/ONNX (`scomp-link export`)
- Agent Skill (SKILL.md) with decision tree, CLI reference, visualization guide, workflow patterns
- 8 new examples (27-34)
- 29 new tests for CLI commands + choose_model branches
- Pre-commit hooks with Ruff linting

### Fixed
- `supervised_text.py` migrated from spaCy v2 to v3 API
- `core.py` handles DataFrames with image/array columns
- `pdf_converter.py` catches OSError from weasyprint

### Changed
- CLI expanded from 13 to 25 commands
- Coverage: 65% → 81% (279 tests + 34 examples)
- Dependencies bumped for Python 3.14 support
- Added optional extras: `[mcp]`, `[serve]`

## 1.2.0 (2026-06-24)

### Added
- **RAWGraphs SVG Charts**: 31 server-side chart functions (alluvial, chord, sankey, treemap, sunburst, etc.)
- **Centralized Colors**: `scomp_link.utils.colors` — single source of truth for all palettes
- `add_rawgraphs_to_report()` method on `ScompLinkHTMLReport`
- `tests/test_rawgraphs.py` (38 tests)
- `examples/example_19_rawgraphs.py`

### Changed
- **Lazy imports (PEP 562)**: `import scomp_link` now takes ~6ms instead of ~5200ms (99.9% faster)
- All public classes loaded on first access via `__getattr__`, not at import time
- All color references now use centralized `colors.py`

### Fixed
- `boxplot()` compatibility with matplotlib 3.9+

## 1.1.4 (2026-06-24)

### Added
- Python 3.14 support (experimental)

### Changed
- Replaced abandoned `pytorch-tabnet` with maintained fork `pytorch-tabnet2`
- Bumped upper bounds: torch <2.13, transformers <6.0, matplotlib <3.11

### Fixed
- CI failure on Python 3.14 due to missing TensorFlow wheels

## 1.1.0 (2026-05-06)

### Added
- CLI with 13 commands
- EnsembleOptimizer (voting/stacking)
- AdvancedCV (LOOCV, Bootstrap)
- AnomalyDetector (IForest, LOF, TabNet, Transformer)
- TimeSeriesAnomalyDetector
- Highcharts visualizations (streamgraph, heatmap, gantt)

## 1.0.0 (2026-02-27)

### Added
- Initial release
- ScompLinkPipeline orchestrator
- RegressorOptimizer with Boruta feature selection
- ClassifierOptimizer
- ModelFactory decision tree
- ScompLinkHTMLReport builder
- Plotly utils (histogram, barchart, linechart, area_chart)
- Explainability (SHAP, LIME)
- Advanced Tuning (Optuna, Halving, EarlyStopping)
- Drift Detection (PSI + KS test)
- Pipeline Persistence (.scomp format)
- Feature Engineering (sklearn-compatible)
- Time Series Forecasting
- Fairness Metrics
- Data Quality Reports
- 18 example scripts
