# Visualization

scomp-link provides three chart libraries and a centralized color system, all accessible through `scomp_link.utils`.

---

## Color Palette (`colors.py`)

All color constants are defined in a single module. Import colors from here — never hardcode hex values.

```python
from scomp_link.utils.colors import PRIMARY, DARK, MAIN
```

### Palettes

| Name | Description | Use Case |
|------|-------------|----------|
| `PRIMARY` | 10 vivid categorical colors | Default palette for all charts |
| `LIGHT` | Lightest variants | Backgrounds, hover states |
| `MEDIUM_LIGHT` | Medium-light | Secondary fills |
| `MEDIUM` | Medium intensity | Accent elements |
| `MEDIUM_DARK` | Medium-dark | Text on light backgrounds |
| `DARK` | Dark variants | Borders, axis lines |
| `DARKEST` | Darkest variants | Headers, strong contrast |

### Theme Constants

| Name | Value | Description |
|------|-------|-------------|
| `MAIN` | `#6E37FA` | Primary brand color |
| `MAIN_LIGHT` | `#9682FF` | Light variant |
| `MAIN_DARK` | `#4614B4` | Dark variant |
| `PRIMARY_JSON` | JSON string | For Highcharts embedding |

---

## RAWGraphs Charts (`rawgraphs/`)

Server-side SVG generation for 31 chart types. Each function returns an embeddable SVG string.

```python
from scomp_link.utils.rawgraphs import barchart, chord_diagram, treemap
from scomp_link.utils.report_html import ScompLinkHTMLReport

svg = barchart(['Q1', 'Q2', 'Q3'], [100, 150, 200], 'Revenue')
report = ScompLinkHTMLReport('Report')
report.add_rawgraphs_to_report(svg, 'Revenue by Quarter')
report.save_html('output.html')
```

### Comparisons

::: scomp_link.utils.rawgraphs.comparisons
    options:
      show_root_heading: false
      members:
        - barchart
        - barchartmultiset
        - barchartstacked
        - piechart
        - radarchart
        - voronoidiagram

### Distributions

::: scomp_link.utils.rawgraphs.distributions
    options:
      show_root_heading: false
      members:
        - beeswarm
        - boxplot
        - violinplot

### Time Series

::: scomp_link.utils.rawgraphs.time_series
    options:
      show_root_heading: false
      members:
        - bumpchart
        - gantt_chart
        - horizongraph
        - linechart
        - slopechart
        - streamgraph

### Correlations

::: scomp_link.utils.rawgraphs.correlations
    options:
      show_root_heading: false
      members:
        - bubblechart
        - contour_plot
        - convex_hull
        - hexagonal_binning
        - matrixplot
        - parallelcoordinates

### Hierarchies

::: scomp_link.utils.rawgraphs.hierarchies
    options:
      show_root_heading: false
      members:
        - circlepacking
        - circular_dendrogram
        - dendrogram
        - sunburst
        - treemap
        - voronoi_treemap

### Networks

::: scomp_link.utils.rawgraphs.networks
    options:
      show_root_heading: false
      members:
        - alluvial_diagram
        - arc_diagram
        - chord_diagram
        - sankey_diagram

---

## Plotly Charts (`plotly_utils.py`)

Interactive Plotly figures for embedding in HTML reports.

```python
from scomp_link.utils.plotly_utils import histogram, barchart, linechart, area_chart
```

::: scomp_link.utils.plotly_utils
    options:
      show_root_heading: false
      members:
        - multiple_histograms
        - histogram
        - barchart
        - linechart
        - area_chart

---

## Highcharts (`highcharts.py`)

JavaScript-rendered charts returned as HTML strings.

```python
from scomp_link.utils.highcharts import streamgraphs, calendar_heatmap, calendar_gantt
```

::: scomp_link.utils.highcharts
    options:
      show_root_heading: false
      members:
        - streamgraphs
        - calendar_heatmap
        - calendar_gantt
