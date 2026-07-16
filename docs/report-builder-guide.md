# scomp-link Report Builder — MCP Guide

Build fully custom, branded HTML reports using AI agents. No code needed — just describe what you want, and the agent builds it step by step.

---

## Installation

### Option 1: Local MCP Server (recommended)

```bash
pip install scomp-link[mcp]
```

Then configure your AI agent:

**Claude Desktop** (`~/Library/Application Support/Claude/claude_desktop_config.json`):
```json
{
  "mcpServers": {
    "scomp-link": {
      "command": "scomp-link",
      "args": ["mcp"]
    }
  }
}
```

**Kiro** (`.kiro/mcp.json`):
```json
{
  "mcpServers": {
    "scomp-link": {
      "command": "scomp-link",
      "args": ["mcp"]
    }
  }
}
```

**Cursor** (Settings → MCP Servers):
```json
{
  "scomp-link": {
    "command": "scomp-link",
    "args": ["mcp"]
  }
}
```

**VS Code Copilot** (`.vscode/mcp.json`):
```json
{
  "servers": {
    "scomp-link": {
      "command": "scomp-link",
      "args": ["mcp"]
    }
  }
}
```

### Option 2: Remote MCP (zero install)

No pip install needed — connect directly to the hosted server:

```json
{
  "mcpServers": {
    "scomp-link": {
      "url": "https://Euribor512-scomp-link.hf.space/sse"
    }
  }
}
```

### Option 3: Docker

```bash
docker run -i jack15121/scomp-link mcp
```

---

## Corporate Branding Setup (one-time)

Set your default report branding so all reports automatically use your corporate style:

```bash
scomp-link init-config
```

This creates `~/.scomp-link/config.yaml`. Edit it with your branding:

```yaml
report:
  font_family: "Arial"
  url_img_logo: "https://www.inter.it/assets/logo.png"
  url_background_header: "https://cdn.inter.it/assets/report-header.jpg"
  description: "Inter Milan Performance Report"
  author: "Inter Data Analytics"
  main_color: "#0068A8"
  light_color: "#4DA6E0"
  dark_color: "#003D6B"
  footer_html: "<footer><strong>FC Internazionale Milano</strong><br>Internal use only. Confidential.<br>Copyright &copy; 2026 Inter. All rights reserved.</footer>"
```

For project-specific overrides, create `.scomp-link.yaml` in the project root:

```bash
scomp-link init-config --local
```

**Precedence:** `.scomp-link.yaml` (local) > `~/.scomp-link/config.yaml` (global) > scomp-link defaults

---

## Report Builder Workflow

### The 6 Tools

| Step | Tool | What it does |
|------|------|-------------|
| 1 | `report_create` | Create a new report session → returns `report_id` |
| 2 | `report_add_section` | Open a collapsible section with a title |
| 3 | `report_add_text` | Add paragraph, title, subtitle, or raw HTML |
| 4 | `report_add_table` | Add an interactive data table (sortable, CSV-exportable) |
| 5 | `report_add_chart` | Add a chart (39 types from 3 engines) |
| 6 | `report_save` | Save to HTML file, close session |

### Basic Example

```
report_create("Inter Season 2026/27 Report")
→ report_id: "a3f8b2c1"

report_add_section("a3f8b2c1", "Executive Summary")
report_add_text("a3f8b2c1", "Strong season: 78 points in Serie A, Champions League QF reached.")
report_add_table("a3f8b2c1", '[{"competition": "Serie A", "matches": 38, "points": 78}, {"competition": "UCL", "matches": 10, "stage": "QF"}]', "Season Overview")

report_add_section("a3f8b2c1", "Trends")
report_add_chart("a3f8b2c1", "plotly", "linechart", '{"dates": ["2024-01-01","2024-02-01","2024-03-01"], "lines": [[6,15,24]], "y_labels": ["Points"]}', "Points Progression")

report_add_section("a3f8b2c1", "Squad Value")
report_add_chart("a3f8b2c1", "rawgraphs", "treemap", '{"data": {"name": "Inter", "children": [{"name": "Serie A", "value": 60}, {"name": "Champions League", "value": 40}]}}', "Squad Value by Department")

report_save("a3f8b2c1", "inter_season_report.html")
```

---

## `report_create` — Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `title` | *required* | Report title (displayed in header) |
| `font_family` | from config | CSS font (e.g. "Arial", "Roboto", "Baloo 2") |
| `url_img_logo` | from config | Public URL to logo image (favicon) |
| `url_background_header` | from config | Header background image URL (ideal: 1920×600px) |
| `description` | from config | HTML meta description |
| `author` | from config | HTML meta author |
| `language` | from config | HTML lang attribute ("en", "it", "de", etc.) |
| `main_color` | from config | Primary accent color hex (headings, links, buttons) |
| `light_color` | from config | Light variant hex (hover states) |
| `dark_color` | from config | Dark variant hex (borders, outlines) |
| `footer_html` | from config | Custom HTML wrapped in `<footer>...</footer>` |

All parameters except `title` default to your config file values. If no config exists, scomp-link built-in defaults are used.

---

## `report_add_text` — Styles

| Style | Renders as |
|-------|-----------|
| `"paragraph"` | `<p>...</p>` — standard body text |
| `"title"` | `<h2>...</h2>` — large section heading |
| `"subtitle"` | `<h3>...</h3>` — medium subheading |
| `"html"` | Raw HTML injected as-is (for custom formatting) |

---

## `report_add_table` — Data Format

Pass a JSON string containing a list of dictionaries (one per row):

```json
[
  {"player": "Lautaro", "goals": 24, "assists": 7},
  {"player": "Thuram", "goals": 15, "assists": 10},
  {"player": "Barella", "goals": 8, "assists": 14}
]
```

Tables are interactive: sortable columns + CSV export button.

---

## `report_add_chart` — The 39 Chart Types

### Engine Selection Guide

| Engine | Best For | Output | Interactivity |
|--------|----------|--------|---------------|
| **plotly** | Dashboards, exploration | HTML | ✅ Hover, zoom, pan |
| **rawgraphs** | Print reports, presentations | SVG | ❌ Static, publication-quality |
| **highcharts** | Time series monitoring | HTML | ✅ Interactive, annotations |

---

### Plotly Charts (4 types)

| Type | Data Format |
|------|------------|
| `histogram` | `{"values": [1.2, 3.4, 5.6, ...], "name": "Column Name"}` |
| `barchart` | `{"categories": ["A","B","C"], "values": [[10,20,30]], "y_axis_titles": ["Points"]}` |
| `linechart` | `{"dates": ["2024-01-01",...], "lines": [[100,120,...]], "y_labels": ["Sales"]}` |
| `area_chart` | Same format as linechart (stacked area) |

**Multiple series** (linechart/area_chart):
```json
{
  "dates": ["2024-01-01", "2024-02-01", "2024-03-01"],
  "lines": [[100, 110, 105], [80, 85, 90]],
  "y_labels": ["Points", "Cost"]
}
```

**Multiple bar groups:**
```json
{
  "categories": ["Q1", "Q2", "Q3"],
  "values": [[100, 150, 120], [80, 90, 110]],
  "y_axis_titles": ["Product A", "Product B"]
}
```

Optional: `"format_date": "%Y-%m"` if dates aren't `%Y-%m-%d`.

---

### RAWGraphs Charts (31 types)

#### Comparisons

| Type | Data Format |
|------|------------|
| `barchart` | `{"categories": [...], "values": [...]}` |
| `barchartmultiset` | `{"categories": [...], "groups": {"A": [...], "B": [...]}}` |
| `barchartstacked` | `{"categories": [...], "groups": {"A": [...], "B": [...]}}` |
| `piechart` | `{"labels": [...], "values": [...]}` |
| `radarchart` | `{"categories": [...], "series": {"Model X": [...], "Model Y": [...]}}` |
| `voronoidiagram` | `{"points": [[x,y], [x,y], ...]}` |

#### Distributions

| Type | Data Format |
|------|------------|
| `beeswarm` | `{"data": [...], "groups": [...]}` |
| `boxplot` | `{"data": [[...],[...]], "labels": [...]}` |
| `violinplot` | `{"data": [[...],[...]], "labels": [...]}` |

#### Time Series

| Type | Data Format |
|------|------------|
| `bumpchart` | `{"ranks": {"A": [1,2,1], "B": [2,1,3]}, "periods": ["Q1","Q2","Q3"]}` |
| `gantt_chart` | `{"tasks": [{"name": "T", "start": "2024-01-01", "end": "2024-02-01", "group": "Phase 1"}]}` |
| `horizongraph` | `{"series": {"Temp": [20,22,19]}, "x_values": ["Mon","Tue","Wed"]}` |
| `linechart` | `{"series": {"Points": [100,120,140]}, "x_values": ["Jan","Feb","Mar"]}` |
| `slopechart` | `{"data": {"Product A": [100, 150], "Product B": [120, 90]}}` |
| `streamgraph` | `{"series": {"A": [10,20,15], "B": [5,15,20]}, "x_values": ["Jan","Feb","Mar"]}` |

#### Correlations

| Type | Data Format |
|------|------------|
| `bubblechart` | `{"x": [...], "y": [...], "size": [...]}` |
| `contour_plot` | `{"x": [...], "y": [...]}` |
| `convex_hull` | `{"x": [...], "y": [...], "groups": [...]}` |
| `hexagonal_binning` | `{"x": [...], "y": [...]}` |
| `matrixplot` | `{"matrix": [[...]], "row_labels": [...], "col_labels": [...]}` |
| `parallelcoordinates` | `{"data": {"col1": [...], "col2": [...]}, "class_column": "col1"}` |

#### Hierarchies

| Type | Data Format |
|------|------------|
| `circlepacking` | `{"data": {"name": "root", "children": [{"name": "A", "value": 30}]}}` |
| `circular_dendrogram` | `{"linkage_matrix": [[...]], "labels": [...]}` |
| `dendrogram` | `{"linkage_matrix": [[...]], "labels": [...]}` |
| `sunburst` | `{"data": {"name": "root", "children": [...]}}` |
| `treemap` | `{"data": {"name": "root", "children": [{"name": "A", "value": 100}]}}` |
| `voronoi_treemap` | `{"data": {"name": "root", "children": [...]}}` |

#### Networks

| Type | Data Format |
|------|------------|
| `alluvial_diagram` | `{"flows": [{"source": "A", "target": "B", "value": 40}]}` |
| `arc_diagram` | `{"nodes": ["A","B","C"], "links": [{"source": 0, "target": 1, "value": 5}]}` |
| `chord_diagram` | `{"matrix": [[0,5,3],[5,0,2],[3,2,0]], "labels": ["A","B","C"]}` |
| `sankey_diagram` | `{"nodes": [{"name": "A", "x": 0}], "links": [{"source": 0, "target": 1, "value": 40}]}` |

---

### Highcharts Charts (3 types)

| Type | Data Format |
|------|------------|
| `streamgraphs` | `{"dates": [...], "series_dict": {"A": [...], "B": [...]}, "annotation": {"event": idx}, "area": true}` |
| `calendar_heatmap` | `{"series_dict": {"2024-01-01": 0.85, "2024-01-02": 0.72}, "min": 0, "max": 1}` |
| `calendar_gantt` | `{"series_dict": [...], "min_date": "2024-01-01", "max_date": "2024-12-31"}` |

---

## Chart Selection Tips

| I want to show... | Use |
|-------------------|-----|
| Distribution of one variable | `plotly` → `histogram` or `rawgraphs` → `violinplot` |
| Compare categories | `rawgraphs` → `barchart` or `radarchart` |
| Trends over time | `plotly` → `linechart` or `highcharts` → `streamgraphs` |
| Part-to-whole | `rawgraphs` → `piechart`, `sunburst`, or `treemap` |
| Correlations | `rawgraphs` → `bubblechart` or `parallelcoordinates` |
| Flows / journeys | `rawgraphs` → `sankey_diagram` or `alluvial_diagram` |
| Hierarchies | `rawgraphs` → `treemap`, `sunburst`, or `circlepacking` |
| Calendar patterns | `highcharts` → `calendar_heatmap` |
| Project timelines / Gantt | `highcharts` → `calendar_gantt` or `rawgraphs` → `gantt_chart` |
| Ranking changes | `rawgraphs` → `bumpchart` or `slopechart` |
| Correlation matrix | `rawgraphs` → `matrixplot` |
| Multi-dimensional | `rawgraphs` → `parallelcoordinates` or `radarchart` |

---

## Quick Setup Prompt for AI Agents

Copy-paste this into your AI agent to enable scomp-link:

> You have access to scomp-link, an ML toolkit with 22 MCP tools. Use them for:
> - **Data profiling**: `describe_data(path)` — always start here
> - **Training**: `train_model(data, target, task)` with optional `tune=true` for Optuna
> - **Validation**: `validate_model(artifact, data, target)` for test evaluation
> - **Reports**: Use the report builder for custom dashboards:
>   1. `report_create(title)` → get report_id
>   2. `report_add_section(id, title)` → structure
>   3. `report_add_chart(id, engine, type, data, title)` → 39 chart types (plotly/rawgraphs/highcharts)
>   4. `report_add_table(id, json_data, title)` → data tables
>   5. `report_save(id, path)` → save HTML
> - **Monitoring**: `detect_drift`, `detect_anomalies`, `check_fairness`
> - **Forecasting**: `forecast_series(data, column, horizon)`
>
> Report branding defaults come from `~/.scomp-link/config.yaml`. Run `scomp-link init-config` to set up.

---

## Full Example: Inter Milan Season Report

```
# 1. Create with corporate branding (auto-loaded from config)
report_create("Inter Milan Season 2026/27 Analysis")
→ report_id: "f9441616"

# 2. Executive Summary
report_add_section("f9441616", "Executive Summary")
report_add_text("f9441616", "Season 2026/27 performance analysis: 78 points in Serie A, Champions League quarter-finals reached.")
report_add_table("f9441616", '[
  {"competition": "Serie A", "matches": 38, "points": 78},
  {"competition": "Champions League", "matches": 10, "points": "QF"},
  {"competition": "Coppa Italia", "matches": 5, "points": "SF"}
]', "Performance by Competition")

# 3. Points Progression (Plotly interactive line chart)
report_add_section("f9441616", "Points Trend")
report_add_chart("f9441616", "plotly", "linechart", '{
  "dates": ["2026-09-01","2026-10-01","2026-11-01","2026-12-01","2027-01-01","2027-02-01","2027-03-01","2027-04-01","2027-05-01"],
  "lines": [[3,9,18,27,33,42,54,66,78],[3,6,9,12,12,15,18,18,21]],
  "y_labels": ["Serie A","Champions League"]
}', "Season Points Progression")

# 4. Squad Value (RAWGraphs static treemap)
report_add_section("f9441616", "Squad Composition")
report_add_chart("f9441616", "rawgraphs", "treemap", '{
  "data": {"name": "Inter", "children": [
    {"name": "Attack", "children": [{"name": "Lautaro", "value": 85}, {"name": "Thuram", "value": 65}]},
    {"name": "Midfield", "children": [{"name": "Barella", "value": 80}, {"name": "Calhanoglu", "value": 60}]},
    {"name": "Defence", "children": [{"name": "Bastoni", "value": 70}, {"name": "Sommer", "value": 15}]}
  ]}
}', "Squad Value by Department")

# 5. Performance Distribution (Plotly histogram)
report_add_section("f9441616", "Player Performance")
report_add_chart("f9441616", "plotly", "histogram", '{
  "values": [10.2,11.5,10.8,12.1,11.3,10.9,11.7,12.4,10.5,11.8,11.2,10.7,11.4,11.9,12.0,11.1,10.6,11.6,12.2,11.0],
  "name": "Sprint Distance (km)"
}', "Sprint Distance Distribution (km)")

# 6. Save
report_save("f9441616", "inter_season_report.html")
→ {"status": "saved", "path": "inter_season_report.html", "size_kb": 58.3}
```

The output is a self-contained HTML file with:
- Corporate header with Inter logo and colors
- Collapsible sections
- Interactive Plotly charts (hover, zoom)
- Publication-quality SVG treemap
- Sortable tables with CSV export
- Custom footer with copyright notice

---

## Python API (alternative to MCP)

If you prefer Python over MCP tools:

```python
from scomp_link.utils.report_html import ScompLinkHTMLReport
from scomp_link.utils.plotly_utils import histogram, linechart
from scomp_link.utils.rawgraphs import treemap

# Create with custom footer
report = ScompLinkHTMLReport(
    title="Season Report",
    main_color="#0068A8",
    footer_html="<footer><strong>FC Internazionale Milano</strong></footer>"
)

# Build
report.open_section("Overview")
report.add_text("Season 2026/27 performance analysis.")
report.add_dataframe(df, "Summary")
report.close_section()

report.open_section("Charts")
fig = linechart(dates, [points_line], "Points Trend")
report.add_graph_to_report(fig, "Points")

svg = treemap(data=hierarchy, title="Squad Value")
report.add_rawgraphs_to_report(svg, "Squad Value")
report.close_section()

# Save
report.save_html("inter_season_report.html")
```

---

## Links

| | |
|-|-|
| 📦 PyPI | [pypi.org/project/scomp-link](https://pypi.org/project/scomp-link/) |
| 🐙 GitHub | [github.com/GiacomoSaccaggi/scomp_link](https://github.com/GiacomoSaccaggi/scomp_link) |
| 🤗 Remote MCP | [huggingface.co/spaces/Euribor512/scomp-link](https://huggingface.co/spaces/Euribor512/scomp-link) |
| 🔧 Smithery | [smithery.ai/servers/giacomosaccaggi/scomp-link](https://smithery.ai/servers/giacomosaccaggi/scomp-link) |
| 🐳 Docker | [hub.docker.com/r/jack15121/scomp-link](https://hub.docker.com/r/jack15121/scomp-link) |
| 📖 Full Docs | [giacomosaccaggi.github.io/scomp_link](https://giacomosaccaggi.github.io/scomp_link/) |
