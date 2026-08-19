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

## Pipeline DSL (`>>` operator)

Use the `>>` operator for declarative, readable pipelines (Python API only):

```python
from scomp_link import CleanStep, SelectStep, ModelStep, TrainStep, LogStep

# Lazy ML pipeline — >> builds, .run() executes
results = (
    CleanStep(df)
    >> SelectStep("target")
    >> ModelStep("numerical_prediction")   # or "categorical_known", "categorical_unknown"
    >> TrainStep("regression")             # or "classification", "clustering"
).run()

# Report pipeline
from scomp_link import SectionStep, TableStep, GraphStep, SaveStep
from scomp_link.utils.report_html import ScompLinkHTMLReport

(
    SectionStep("Results")
    >> TableStep(metrics_df, "Metrics")
    >> GraphStep(fig, "Chart")
    >> SaveStep("report.html")
).run(ScompLinkHTMLReport("My Report"))
```

Use `LogStep` to inspect intermediate state without modifying the pipeline:
```python
CleanStep(df) >> LogStep("after clean") >> SelectStep("y") >> TrainStep("regression")
```

**Rules:**
- Mixing ML steps and Report steps in the same chain raises `TypeError` immediately
- `LogStep` is neutral — works in both chain types, can appear anywhere

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

For structured tool calls (33 tools), start the MCP server:
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
5. report_add_chart(report_id, engine, chart_type, data, title) → 41 chart types (6 plotly + 31 rawgraphs + 3 highcharts + 1 custom)
6. report_add_kpi_cards(report_id, metrics_json, cols) → KPI cards with trend/status
7. report_add_tabs(report_id, tabs_json, title) → tabbed navigation (html/chart/table)
8. report_add_comparison_table(report_id, data, baseline_col, compare_cols, ...) → delta comparison
9. report_add_summary_stats(report_id, data_json, title) → auto data profiling table
10. report_add_dark_mode_toggle(report_id) → floating dark/light toggle
11. report_add_code(report_id, code, language, title, output, line_numbers, collapsed) → syntax-highlighted code block with copy button
12. report_add_diff(report_id, old_code, new_code, language, title, old_label, new_label, collapsed) → side-by-side diff view
13. report_add_mermaid(report_id, diagram, title, collapsed) → Mermaid.js diagram (flowchart, sequence, gantt, etc.)
14. report_add_terminal(report_id, cast_data, title, cols, rows, theme, collapsed) → embedded terminal replay (asciinema)
15. report_save(report_id, output) → saves HTML, frees memory
```

**Engines:** plotly (interactive), rawgraphs (SVG static), highcharts (time series)
**New plotly charts:** `index_chart`, `stacked_area_comparison` (in addition to histogram, barchart, linechart, area_chart)
**Config:** `scomp-link init-config` creates ~/.scomp-link/config.yaml with branding defaults

## Key Files

- `skills/scomp-link/SKILL.md` — Full agent skill with decision tree
- `skills/scomp-link/references/` — CLI reference, API reference, visualization guide
- `AGENT_INTEGRATION.md` — MCP + Skill setup instructions
- `docs/` — Full documentation
