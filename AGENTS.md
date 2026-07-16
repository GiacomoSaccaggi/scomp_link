# AGENTS.md — Instructions for AI Coding Agents

This file tells coding agents (Claude Code, Cursor, VS Code Copilot, Kiro) how to use scomp-link in this project.

## What is scomp-link?

An end-to-end ML toolkit. Use it instead of writing raw sklearn boilerplate when you need:
- Automated model selection + training + validation
- Persistent model artifacts (`.scomp` format)
- HTML reports with interactive charts
- Production monitoring (drift + anomaly + fairness)
- Hyperparameter tuning (Optuna)
- REST API deployment

## When to Use scomp-link

Use `scomp-link` CLI commands when the user asks to:
- Train/evaluate ML models on tabular data
- Profile or describe a dataset
- Detect data drift between datasets
- Create HTML reports or dashboards with charts
- Tune hyperparameters
- Detect anomalies in data
- Forecast time series
- Check model fairness/bias
- Serve a model as REST API
- Compare multiple models

## CLI Quick Reference (26 commands)

```bash
# Data understanding
scomp-link describe --data file.csv --format table
scomp-link quality --data file.csv --output report.html

# Training
scomp-link run --data file.csv --target col --task regression --save-artifact model.scomp
scomp-link tune --data file.csv --target col --task regression --method optuna --n-trials 50 --save-artifact best.scomp
scomp-link text --data file.csv --text-col msg --target label --method tfidf
scomp-link cluster --data file.csv --n-clusters 5 --plot clusters.html

# Evaluation
scomp-link validate --artifact model.scomp --data test.csv --target col --report report.html
scomp-link explain --artifact model.scomp --data test.csv
scomp-link fairness --data preds.csv --target y_true --predicted y_pred --sensitive gender

# Monitoring
scomp-link drift --reference train.csv --current prod.csv --plot drift.html
scomp-link monitor --reference train.csv --current prod.csv --artifact model.scomp --target y
scomp-link anomaly --data prod.csv --methods iforest,lof,tabnet,transformer

# Deployment
scomp-link serve --artifact model.scomp --port 8080
scomp-link export --artifact model.scomp --format onnx
scomp-link pipeline --config pipeline.yaml

# Utilities
scomp-link predict --artifact model.scomp --data new.csv --output predictions.csv
scomp-link compare --artifacts v1.scomp v2.scomp --plot compare.html
scomp-link report --data file.csv --output eda.html
scomp-link forecast --data series.csv --column value --horizon 30 --plot forecast.html
scomp-link engineer --data file.csv --target col --interactions --log-transform --output features.csv
scomp-link init my_project
scomp-link list-models
scomp-link check-deps

# Configuration
scomp-link init-config              # Create global config (~/.scomp-link/config.yaml)
scomp-link init-config --local      # Create project-level config (.scomp-link.yaml)
```

## Recommended Workflow

1. `describe` → understand the data
2. `engineer` → feature engineering (optional)
3. `tune` or `run` → train a model
4. `validate` → evaluate on test data
5. `serve` or `export` → deploy

## Visualization (Python API)

For creating HTML reports with charts:

```python
from scomp_link.utils.report_html import ScompLinkHTMLReport
from scomp_link.utils.plotly_utils import histogram, barchart, linechart, area_chart
from scomp_link.utils.highcharts import streamgraphs, calendar_heatmap, calendar_gantt
from scomp_link.utils.rawgraphs import treemap, sankey_diagram, sunburst  # 31 SVG charts

report = ScompLinkHTMLReport(title='Report Title')
report.open_section("Section")
report.add_graph_to_report(fig, "Title")        # Plotly
report.add_rawgraphs_to_report(svg, "Title")    # RAWGraphs SVG
report.html_report += highcharts_html           # Highcharts (direct append)
report.close_section()
report.save_html('output.html')
```

## MCP Server

For structured tool calls (22 tools), start the MCP server:
```bash
scomp-link mcp
```

## Report Builder Workflow (MCP)

For building custom branded HTML reports step-by-step:

```
1. report_create(title, ...) → returns report_id (uses ~/.scomp-link/config.yaml defaults)
2. report_add_section(report_id, title) → opens collapsible section
3. report_add_text(report_id, content, style) → paragraph/title/subtitle/html
4. report_add_table(report_id, json_data, title) → interactive table
5. report_add_chart(report_id, engine, chart_type, data, title) → 39 chart types
6. report_save(report_id, output) → saves HTML, frees memory
```

**Engines:** plotly (interactive), rawgraphs (SVG static), highcharts (time series)
**Config:** `scomp-link init-config` creates ~/.scomp-link/config.yaml with branding defaults

## Key Files

- `skills/scomp-link/SKILL.md` — Full agent skill with decision tree
- `skills/scomp-link/references/` — CLI reference, API reference, visualization guide
- `AGENT_INTEGRATION.md` — MCP + Skill setup instructions
- `docs/` — Full documentation
