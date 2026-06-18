# Utils

Logging, HTML report generation, charts, and PDF conversion.

## Modules

| Module | Class/Function | Description |
|--------|---------------|-------------|
| `logger.py` | `get_logger()`, `set_verbosity()` | Configurable logging (silent/warning/info/debug) |
| `report_html.py` | `ScompLinkHTMLReport` | Stateful HTML report builder with Plotly embedding |
| `plotly_utils.py` | `histogram`, `barchart`, `linechart`, `area_chart` | Chart generation utilities |
| `highcharts.py` | `streamgraphs`, `calendar_heatmap`, `calendar_gantt` | Highcharts-based visualizations |
| `pdf_converter.py` | — | HTML to PDF via WeasyPrint/Playwright |
| `decorators.py` | — | Validation and utility decorators |

## Usage

```python
from scomp_link import set_verbosity

set_verbosity("silent")   # suppress all library output
set_verbosity("info")     # default (progress messages)
set_verbosity("debug")    # verbose
```
