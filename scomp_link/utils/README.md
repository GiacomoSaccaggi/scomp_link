# Utils

Logging, HTML report generation, charts, and PDF conversion.

## Modules

| Module | Class/Function | Description |
|--------|---------------|-------------|
| `logger.py` | `get_logger()`, `set_verbosity()` | Configurable logging (silent/warning/info/debug) |
| `report_html.py` | `ScompLinkHTMLReport` | Stateful HTML report builder with Plotly embedding |
| `plotly_utils.py` | `histogram`, `barchart`, `linechart`, `area_chart` | Plotly chart generation utilities |
| `highcharts.py` | `streamgraphs`, `calendar_heatmap`, `calendar_gantt` | Highcharts-based visualizations |
| `rawgraphs/` | 31 chart functions | Server-side SVG chart generation (see [rawgraphs/README.md](rawgraphs/README.md)) |
| `pdf_converter.py` | — | HTML to PDF via WeasyPrint/Playwright |
| `decorators.py` | — | Validation and utility decorators |

## Chart Libraries Comparison

| Library | Output Format | Requires Browser | Use Case |
|---------|--------------|-----------------|----------|
| `plotly_utils` | Plotly Figure (interactive JS) | No (embedded JS) | Interactive exploration |
| `highcharts` | HTML + JS string | No (embedded JS) | Streamgraphs, calendars, Gantt |
| `rawgraphs/` | SVG string (static) | No | Publication-quality static charts (31 types) |

## Usage

```python
from scomp_link import set_verbosity

set_verbosity("silent")   # suppress all library output
set_verbosity("info")     # default (progress messages)
set_verbosity("debug")    # verbose
```

### Adding charts to a report

```python
from scomp_link.utils.report_html import ScompLinkHTMLReport
from scomp_link.utils.plotly_utils import histogram
from scomp_link.utils.rawgraphs import chord_diagram, treemap

report = ScompLinkHTMLReport('My Report')

# Plotly chart
fig = histogram(data, 'Distribution')
report.add_graph_to_report(fig, 'Histogram')

# RAWGraphs SVG chart
svg = chord_diagram([[0,5,3],[5,0,4],[3,4,0]], ['A','B','C'], 'Flows')
report.add_rawgraphs_to_report(svg, 'Chord Diagram')

report.save_html('report.html')
```
