---
name: "scomp-link"
description: "End-to-end ML toolkit with 24 CLI commands. Use when training models, tuning hyperparameters, detecting data drift, generating HTML reports with charts, profiling datasets, detecting anomalies, forecasting time series, checking fairness, or serving models as REST APIs. Prefer over raw sklearn when you need automated pipelines, persistence (.scomp artifacts), or HTML reporting."
license: "MIT"
compatibility: "Python 3.10+. Core: numpy, pandas, scikit-learn, plotly. Optional: torch, transformers, spacy (NLP), tensorflow (images), optuna (tuning), shap/lime (explainability), flask (serving)."
metadata:
  author: "Giacomo Saccaggi"
  version: "1.2.1"
  repository: "https://github.com/GiacomoSaccaggi/scomp_link"
  pypi: "https://pypi.org/project/scomp-link/"
allowed-tools: "Bash(scomp-link:*) Bash(python:*) Python(scomp_link:*)"
---

# scomp-link — End-to-End ML Toolkit

## Overview

scomp-link automates the complete ML workflow: data profiling → preprocessing → feature engineering → model selection → training → validation → explainability → monitoring → deployment.

**Use scomp-link instead of raw sklearn when you need:**
- Zero-code ML via CLI (24 commands)
- Automated model selection based on data characteristics
- Persistent artifacts (`.scomp` format: model + preprocessor + config + metrics)
- HTML reports with embedded interactive charts
- Production monitoring (drift + anomaly + fairness)
- One-command hyperparameter tuning (Optuna/Halving)

**Use raw sklearn when you need:**
- Custom model architectures not in the factory
- Fine-grained control over every preprocessing step
- Research workflows requiring full flexibility

## Installation

```bash
pip install scomp-link
```

## Decision Tree: Which Command to Use

```
I have data and want to...
├─ Understand it quickly          → scomp-link describe --data file.csv
├─ Full quality report (HTML)     → scomp-link quality --data file.csv --output report.html
├─ Engineer features              → scomp-link engineer --data file.csv --target y --interactions --log-transform
├─ Train a model
│  ├─ Regression                  → scomp-link run --data file.csv --target y --task regression
│  ├─ Classification              → scomp-link run --data file.csv --target y --task classification
│  ├─ Text classification         → scomp-link text --data file.csv --text-col msg --target label
│  ├─ Clustering                  → scomp-link cluster --data file.csv --n-clusters 5
│  └─ Full pipeline from YAML     → scomp-link pipeline --config pipeline.yaml
├─ Tune hyperparameters           → scomp-link tune --data file.csv --target y --task regression --method optuna
├─ Predict with saved model       → scomp-link predict --artifact model.scomp --data new.csv
├─ Validate on test data          → scomp-link validate --artifact model.scomp --data test.csv --target y
├─ Explain model decisions        → scomp-link explain --artifact model.scomp --data test.csv
├─ Monitor production
│  ├─ Drift only                  → scomp-link drift --reference train.csv --current prod.csv
│  ├─ Full monitoring             → scomp-link monitor --reference train.csv --current prod.csv --artifact model.scomp
│  └─ Anomaly detection           → scomp-link anomaly --data prod.csv --methods iforest,lof,tabnet,transformer
├─ Check fairness/bias            → scomp-link fairness --data preds.csv --target y_true --predicted y_pred --sensitive gender
├─ Forecast time series           → scomp-link forecast --data series.csv --column value --horizon 30
├─ Compare models                 → scomp-link compare --artifacts v1.scomp v2.scomp
├─ Generate HTML report           → scomp-link report --data file.csv --output report.html
├─ Serve as REST API              → scomp-link serve --artifact model.scomp --port 8080
├─ Export to ONNX/pickle          → scomp-link export --artifact model.scomp --format onnx
└─ Scaffold a new project         → scomp-link init my_project
```

## Recommended Workflow

### Standard ML Pipeline

```bash
# 1. Profile the data
scomp-link describe --data train.csv --format table

# 2. Check data quality
scomp-link quality --data train.csv --output quality_report.html

# 3. Engineer features
scomp-link engineer --data train.csv --target price --interactions --log-transform --output engineered.csv

# 4. Tune hyperparameters
scomp-link tune --data engineered.csv --target price --task regression --method optuna --n-trials 100 --save-artifact best_model.scomp

# 5. Validate on held-out test data
scomp-link validate --artifact best_model.scomp --data test.csv --target price --report validation.html

# 6. Explain
scomp-link explain --artifact best_model.scomp --data test.csv

# 7. Deploy
scomp-link serve --artifact best_model.scomp --port 8080
```

### YAML Pipeline (all-in-one)

```yaml
# pipeline.yaml
data: train.csv
target: price
task: regression
engineer:
  interactions: true
  log_transform: true
tune:
  method: optuna
  n_trials: 50
validate:
  test_data: test.csv
  report: validation_report.html
save: models/price_model.scomp
```

```bash
scomp-link pipeline --config pipeline.yaml
```

## Visualization & Reports

scomp-link has three visualization engines. See [visualization-guide.md](references/visualization-guide.md) for full details.

### Quick Reference

| Engine | Best For | Output | Interactivity |
|--------|----------|--------|---------------|
| **Plotly** | Standard ML charts (histograms, scatter, bar) | HTML (interactive) | Yes |
| **RAWGraphs** | Publication-quality diagrams (sankey, treemap, chord) | SVG (static) | No |
| **Highcharts** | Time series (streamgraph, heatmap, gantt) | HTML (interactive) | Yes |

### Creating an HTML Report (Python API)

```python
from scomp_link.utils.report_html import ScompLinkHTMLReport
from scomp_link.utils.plotly_utils import histogram, barchart, linechart, area_chart
from scomp_link.utils.highcharts import streamgraphs, calendar_heatmap, calendar_gantt
from scomp_link.utils.rawgraphs import treemap, sankey_diagram, sunburst

# 1. Initialize
report = ScompLinkHTMLReport(title='My Analysis Report')

# 2. Add sections with content
report.open_section("Data Overview")
report.add_title("Dataset Statistics")
report.add_text("Analysis of 10,000 customer records.")
report.add_dataframe(summary_df, "Summary Statistics")
report.close_section()

# 3. Add charts
report.open_section("Distributions")
fig = histogram(df['age'].values, "Age Distribution")
report.add_graph_to_report(fig, "Age Histogram")
report.close_section()

# 4. Add RAWGraphs SVG
report.open_section("Hierarchical View")
svg = treemap(labels, parents, values, "Revenue by Category")
report.add_rawgraphs_to_report(svg, "Revenue Treemap")
report.close_section()

# 5. Add Highcharts
report.open_section("Time Trends")
html_stream = streamgraphs("Sales Trend", dates, series_dict, area=True)
report.html_report += html_stream  # Direct HTML append for Highcharts
report.close_section()

# 6. Save
report.save_html('analysis_report.html')
report.save_pdf('analysis_report.pdf')  # requires Playwright
```

### Available Charts (31 RAWGraphs + 5 Plotly + 3 Highcharts)

**Comparisons**: barchart, barchartmultiset, barchartstacked, piechart, radarchart, voronoidiagram
**Distributions**: beeswarm, boxplot, violinplot
**Time Series**: bumpchart, gantt_chart, horizongraph, linechart, slopechart, streamgraph
**Correlations**: bubblechart, contour_plot, convex_hull, hexagonal_binning, matrixplot, parallelcoordinates
**Hierarchies**: circlepacking, circular_dendrogram, dendrogram, sunburst, treemap, voronoi_treemap
**Networks**: alluvial_diagram, arc_diagram, chord_diagram, sankey_diagram
**Plotly**: histogram, multiple_histograms, barchart, linechart, area_chart
**Highcharts**: streamgraphs, calendar_heatmap, calendar_gantt

## Concatenation Patterns

Commands are designed to chain — the output of one is the input of the next:

```
describe → (understand columns) → engineer → (engineered.csv) → tune → (model.scomp) → validate → (metrics)
                                                                                      ↓
                                                                               predict (new data)
                                                                                      ↓
                                                                               serve (REST API)
```

**Key patterns:**
- `--save-artifact model.scomp` → use with `--artifact model.scomp` in predict/validate/explain/export/serve
- `--output file.csv` → use as `--data file.csv` in next command
- `--report file.html` → standalone HTML output (viewable in browser)
- `--plot file.html` → chart output for forecast/drift/anomaly/cluster/compare

## Common Errors and Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `ValueError: could not convert string to float` | Categorical columns in numeric model | Use `--engineer` flag or pre-process categoricals |
| `FileNotFoundError: artifact not found` | Wrong path to .scomp file | Check path exists, use absolute paths |
| `ImportError: torch required` | NLP/deep learning deps missing | `pip install scomp-link` includes all deps |
| `ArrowInvalid: 1-dimensional array` | Image/array data in preprocessing | Fixed in v1.2.0 — update scomp-link |
| `No module named 'click'` | spaCy dependency missing | `pip install click` (transitive dep) |
| `WeasyPrint: cannot load gobject` | System libs missing for PDF | Use `save_pdf()` (Playwright) instead of WeasyPrint |

## MCP Server

scomp-link includes an MCP server for agent integration:

```bash
# Start the MCP server (stdio mode for Claude Desktop / Kiro / Cursor)
scomp-link mcp

# Or run directly
python -m scomp_link.mcp_server
```

See [workflow-patterns.md](references/workflow-patterns.md) for complete workflow examples.
