# RAWGraphs SVG Charts

Server-side SVG chart generation inspired by [RAWGraphs](https://rawgraphs.io/). Each function returns an embeddable SVG string — no browser or JavaScript required.

## Quick Start

```python
from scomp_link.utils.rawgraphs import barchart, chord_diagram, treemap
from scomp_link.utils.report_html import ScompLinkHTMLReport

# Generate SVG
svg = barchart(['Q1', 'Q2', 'Q3', 'Q4'], [120, 180, 150, 210], 'Revenue')

# Embed in a report
report = ScompLinkHTMLReport('My Report')
report.add_rawgraphs_to_report(svg, 'Quarterly Revenue')
report.save_html('output.html')
```

## Available Charts (31 functions)

### Comparisons (`comparisons.py`)

| Function | Description | Key Parameters |
|----------|-------------|----------------|
| `barchart` | Vertical bar chart | `categories`, `values` |
| `barchartmultiset` | Grouped bars side by side | `categories`, `groups: dict` |
| `barchartstacked` | Stacked bars | `categories`, `groups: dict` |
| `piechart` | Pie chart with labels | `labels`, `values` |
| `radarchart` | Radar/spider chart | `categories`, `series: dict` |
| `voronoidiagram` | Voronoi tessellation from 2D points | `points: list[tuple]` |

### Distributions (`distributions.py`)

| Function | Description | Key Parameters |
|----------|-------------|----------------|
| `beeswarm` | Jittered strip plot | `data: list`, `groups: list` |
| `boxplot` | Box-and-whisker | `data: list[list]`, `labels` |
| `violinplot` | Violin density plot | `data: list[list]`, `labels` |

### Time Series (`time_series.py`)

| Function | Description | Key Parameters |
|----------|-------------|----------------|
| `bumpchart` | Ranking changes over time | `ranks: dict`, `periods` |
| `gantt_chart` | Project timeline | `tasks: list[dict]` with `name`, `start`, `end`, `group` |
| `horizongraph` | Layered band chart | `series: dict`, `x_values` |
| `linechart` | Multi-line chart | `series: dict`, `x_values` |
| `slopechart` | Before/after comparison | `data: dict` with `[start, end]` values |
| `streamgraph` | Stacked area (symmetric baseline) | `series: dict`, `x_values` |

### Correlations (`correlations.py`)

| Function | Description | Key Parameters |
|----------|-------------|----------------|
| `bubblechart` | Scatter with sized bubbles | `x`, `y`, `size` |
| `contour_plot` | 2D kernel density contour | `x`, `y` |
| `convex_hull` | Scatter + convex hull per group | `x`, `y`, `groups` |
| `hexagonal_binning` | Hexbin density | `x`, `y`, `gridsize` |
| `matrixplot` | Heatmap matrix | `matrix: list[list]`, `row_labels`, `col_labels` |
| `parallelcoordinates` | Parallel coordinates | `data: dict`, `class_column` |

### Hierarchies (`hierarchies.py`)

| Function | Description | Key Parameters |
|----------|-------------|----------------|
| `circlepacking` | Nested circles | `data: dict` (hierarchical) |
| `circular_dendrogram` | Polar dendrogram | `linkage_matrix`, `labels` |
| `dendrogram` | Linear dendrogram | `linkage_matrix`, `labels` |
| `sunburst` | Nested rings | `data: dict` (hierarchical) |
| `treemap` | Area-proportional rectangles | `data: dict` (hierarchical) |
| `voronoi_treemap` | Weighted Voronoi cells | `data: dict` (hierarchical) |

### Networks (`networks.py`)

| Function | Description | Key Parameters |
|----------|-------------|----------------|
| `alluvial_diagram` | Flow between categories | `flows: list[dict]` with `source`, `target`, `value` |
| `arc_diagram` | Nodes on line with arcs | `nodes: list`, `links: list[dict]` |
| `chord_diagram` | Circular flow matrix | `matrix: list[list]`, `labels` |
| `sankey_diagram` | Multi-column flow | `nodes: list[dict]`, `links: list[dict]` |

## Common Parameters

All functions share these optional parameters:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `title` | `str` | `''` | Chart title |
| `width` | `int` | 600–900 | SVG width in pixels |
| `height` | `int` | 400–700 | SVG height in pixels |
| `colors` | `list` | `None` | Custom color palette (defaults to scomp-link palette) |

## Data Formats

### Hierarchical data (circlepacking, sunburst, treemap, voronoi_treemap)

```python
data = {
    'name': 'root',
    'children': [
        {'name': 'Group A', 'value': 30},
        {'name': 'Group B', 'children': [
            {'name': 'Sub 1', 'value': 15},
            {'name': 'Sub 2', 'value': 10},
        ]},
    ]
}
```

### Linkage matrix (dendrogram, circular_dendrogram)

```python
from scipy.cluster.hierarchy import linkage
import numpy as np

X = np.random.rand(10, 4)  # 10 samples, 4 features
Z = linkage(X, method='ward')
svg = dendrogram(Z, labels=['s1', 's2', ..., 's10'])
```

### Flow data (alluvial_diagram, sankey_diagram)

```python
# Alluvial
flows = [
    {'source': 'Homepage', 'target': 'Products', 'value': 40},
    {'source': 'Homepage', 'target': 'Blog', 'value': 15},
]

# Sankey (nodes need 'x' for column position)
nodes = [{'name': 'Solar', 'x': 0}, {'name': 'Grid', 'x': 1}, {'name': 'Home', 'x': 2}]
links = [{'source': 0, 'target': 1, 'value': 40}, {'source': 1, 'target': 2, 'value': 35}]
```

## Dependencies

Uses only existing scomp-link dependencies:
- **matplotlib** — primary SVG rendering engine
- **numpy** — geometry calculations
- **scipy** — Voronoi tessellation, hierarchical clustering, KDE

## Running Demos

Each file has a `__main__` demo block:

```bash
cd scomp_link/utils/rawgraphs
python -m scomp_link.utils.rawgraphs.comparisons      # → tmp/demo_comparisons.html
python -m scomp_link.utils.rawgraphs.distributions    # → tmp/demo_distributions.html
python -m scomp_link.utils.rawgraphs.time_series      # → tmp/demo_time_series.html
python -m scomp_link.utils.rawgraphs.correlations     # → tmp/demo_correlations.html
python -m scomp_link.utils.rawgraphs.hierarchies      # → tmp/demo_hierarchies.html
python -m scomp_link.utils.rawgraphs.networks         # → tmp/demo_networks.html
```
