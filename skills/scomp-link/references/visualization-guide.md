# Visualization Guide

scomp-link provides 39 chart types across three engines plus an HTML report builder.

## HTML Report Builder Pattern

```python
from scomp_link.utils.report_html import ScompLinkHTMLReport

# Initialize (Highcharts/Plotly JS included automatically)
report = ScompLinkHTMLReport(
    title='Report Title',
    main_color='#6E37FA',    # Optional: primary theme color
    light_color='#9682FF',   # Optional: light variant
    dark_color='#4614B4',    # Optional: dark variant
)

# Build content
report.open_section("Section Name")       # Collapsible section start
report.add_title("Heading")               # <h2> heading
report.add_text("Paragraph text")         # <p> paragraph
report.add_dataframe(df, "Table Name")    # Styled HTML table with CSV download
report.add_graph_to_report(fig, "Title")  # Plotly figure (interactive)
report.add_matplotlib_graph_to_report(fig, "Title")  # Matplotlib (static image)
report.add_rawgraphs_to_report(svg, "Title")  # RAWGraphs SVG chart
report.add_image_to_report("path.png", "Title")  # Local image
report.add_many_plots_with_selection_box_to_report(figs_dict, "Title")  # Combobox selector
report.html_report += html_string         # Direct HTML append (for Highcharts)
report.close_section()                    # Collapsible section end

# Save
report.save_html('output.html')           # Self-contained HTML file
report.save_pdf('output.pdf')             # PDF via Playwright (headless Chrome)
```

## Plotly Utils (5 functions)

```python
from scomp_link.utils.plotly_utils import histogram, multiple_histograms, barchart, linechart, area_chart
```

### `histogram(values, name, h=600)`
- `values`: array-like of floats
- `name`: str — title/axis label
- `h`: int — height in pixels
- Returns: plotly.graph_objects.Figure

### `multiple_histograms(variable_float_for_distribution, ...)`
- Multiple distributions overlaid

### `barchart(categories, metric_values_list, x_axis_title, y_axis_titles=None, order='asc', ...)`
- `categories`: list of str — x-axis labels
- `metric_values_list`: list of float — bar heights
- `order`: 'asc' | 'desc' | None
- `metric_values_line_list`: optional secondary y-axis (line overlay)
- `percentage_y`: bool — format y as percentage
- Returns: plotly.graph_objects.Figure

### `linechart(date_list, lines, title_text, x_label, y_labels, format_date="%Y-%m-%d")`
- `date_list`: list of date strings
- `lines`: dict `{series_name: [values]}`
- Returns: plotly.graph_objects.Figure

### `area_chart(date_list, lines, title_text, x_label, y_labels, format_date="%Y-%m-%d")`
- Same as linechart but filled area
- Returns: plotly.graph_objects.Figure

## Highcharts (3 functions)

```python
from scomp_link.utils.highcharts import streamgraphs, calendar_heatmap, calendar_gantt
```

### `streamgraphs(title, dates, series_dict, annotation=None, area=True)`
- `title`: str
- `dates`: list of str (x-axis categories, e.g. ['2024-01', '2024-02', ...])
- `series_dict`: dict `{series_name: [int_values]}` — each value list same length as dates
- `annotation`: dict `{label: int_index}` — annotations at specific x positions (optional)
- `area`: bool — True=stacked area, False=symmetric streamgraph
- Returns: HTML string (append to `report.html_report`)

### `calendar_heatmap(title, series_dict, min=0, max=1)`
- `title`: str
- `series_dict`: dict `{"yyyy-mm-dd": float_value}` — daily values (typically 28-42 days)
- `min`: float — color scale minimum
- `max`: float — color scale maximum
- Returns: HTML string

### `calendar_gantt(title, series_dict, min_date, max_date, colors=None)`
- `title`: str
- `series_dict`: list of dicts with structure:
  ```python
  [
      {
          'name': 'Phase Name',
          'data': [
              {'name': 'Task', 'id': 'task1',
               'start': "Date.UTC(2025, 5, 1)",  # JS Date.UTC format
               'end': "Date.UTC(2025, 5, 14)",
               'completed': "{ amount: 0.8 }"},  # Optional progress
              {'name': 'Milestone', 'id': 'ms1',
               'start': "Date.UTC(2025, 5, 14)",
               'end': "Date.UTC(2025, 5, 14)",
               'milestone': 'true'},
          ]
      }
  ]
  ```
- `min_date`: str "yyyy-mm-dd"
- `max_date`: str "yyyy-mm-dd"
- Returns: HTML string

## RAWGraphs SVG Charts (31 functions)

All RAWGraphs functions return SVG strings. Embed with `report.add_rawgraphs_to_report(svg, title)`.

```python
from scomp_link.utils.rawgraphs import (
    # Comparisons
    barchart, barchartmultiset, barchartstacked, piechart, radarchart, voronoidiagram,
    # Distributions
    beeswarm, boxplot, violinplot,
    # Time Series
    bumpchart, gantt_chart, horizongraph, linechart, slopechart, streamgraph,
    # Correlations
    bubblechart, contour_plot, convex_hull, hexagonal_binning, matrixplot, parallelcoordinates,
    # Hierarchies
    circlepacking, circular_dendrogram, dendrogram, sunburst, treemap, voronoi_treemap,
    # Networks
    alluvial_diagram, arc_diagram, chord_diagram, sankey_diagram,
)
```

### Comparisons

**`barchart(categories, values, title, width=800, height=500)`**
- `categories`: list[str]
- `values`: list[float]

**`barchartmultiset(categories, series_dict, title, width=800, height=500)`**
- `series_dict`: dict `{series_name: [values]}`

**`barchartstacked(categories, series_dict, title, width=800, height=500)`**
- Same as multiset but stacked

**`piechart(categories, values, title, width=500, height=500)`**

**`radarchart(categories, series_dict, title, width=600, height=600)`**
- `series_dict`: dict `{series_name: [values]}` — values on same scale per category

**`voronoidiagram(points, labels, title, width=800, height=600)`**
- `points`: list of (x, y) tuples
- `labels`: list[str]

### Distributions

**`beeswarm(groups, values, title, width=800, height=400)`**
- `groups`: list[str] — group per point
- `values`: list[float] — value per point

**`boxplot(groups_dict, title, width=800, height=400)`**
- `groups_dict`: dict `{group_name: [values]}`

**`violinplot(groups_dict, title, width=800, height=400)`**
- Same as boxplot

### Time Series

**`bumpchart(time_points, rankings_dict, title, width=900, height=500)`**
- `time_points`: list[str] — x-axis labels
- `rankings_dict`: dict `{entity: [rank_per_timepoint]}`

**`gantt_chart(tasks, title, width=900, height=400)`**
- `tasks`: list of `{'name': str, 'start': float, 'end': float, 'group': str}`

**`horizongraph(dates, series_dict, title, width=900, height=300)`**
- `series_dict`: dict `{series_name: [values]}`

**`linechart(dates, series_dict, title, width=900, height=400)`**
- `dates`: list[str]
- `series_dict`: dict `{series_name: [values]}`

**`slopechart(labels, start_values, end_values, title, start_label, end_label, width=600, height=500)`**

**`streamgraph(dates, series_dict, title, width=900, height=400)`**

### Correlations

**`bubblechart(x, y, sizes, labels, title, width=800, height=600)`**
- `x`, `y`: list[float]
- `sizes`: list[float] — bubble radius
- `labels`: list[str]

**`contour_plot(x, y, title, width=800, height=600, n_levels=10)`**

**`convex_hull(groups_dict, title, width=800, height=600)`**
- `groups_dict`: dict `{group: [(x,y), ...]}`

**`hexagonal_binning(x, y, title, width=800, height=600, gridsize=20)`**

**`matrixplot(matrix, row_labels, col_labels, title, width=700, height=700)`**
- `matrix`: 2D list or numpy array

**`parallelcoordinates(df, group_col, title, width=900, height=400)`**
- `df`: pandas DataFrame
- `group_col`: str — column to color by

### Hierarchies

**`circlepacking(labels, parents, values, title, width=700, height=700)`**
- `labels`: list[str] — node names
- `parents`: list[str] — parent of each node ("" for root)
- `values`: list[float] — size of each node

**`circular_dendrogram(labels, parents, title, width=700, height=700)`**

**`dendrogram(labels, parents, title, width=900, height=500)`**

**`sunburst(labels, parents, values, title, width=700, height=700)`**

**`treemap(labels, parents, values, title, width=900, height=600)`**

**`voronoi_treemap(labels, parents, values, title, width=700, height=700)`**

### Networks

**`alluvial_diagram(flows, title, width=900, height=500)`**
- `flows`: list of `{'source': str, 'target': str, 'value': float}`

**`arc_diagram(nodes, links, title, width=900, height=400)`**
- `nodes`: list[str]
- `links`: list of `{'source': int, 'target': int, 'value': float}`

**`chord_diagram(matrix, labels, title, width=700, height=700)`**
- `matrix`: square 2D array (flow between nodes)
- `labels`: list[str]

**`sankey_diagram(nodes, links, title, width=900, height=500)`**
- `nodes`: list[str]
- `links`: list of `{'source': int, 'target': int, 'value': float}`

## Chart Selection Guide

| I have... | Use this chart |
|-----------|---------------|
| Categories + values | `barchart`, `piechart` |
| Categories + multiple series | `barchartmultiset`, `barchartstacked`, `radarchart` |
| Numeric distribution | `histogram` (Plotly), `boxplot`, `violinplot`, `beeswarm` |
| Time series (single) | `linechart`, `area_chart` (Plotly) |
| Time series (multi-series) | `streamgraphs` (Highcharts), `streamgraph` (RAWGraphs) |
| Rankings over time | `bumpchart`, `slopechart` |
| Project timeline | `calendar_gantt` (Highcharts), `gantt_chart` (RAWGraphs) |
| Daily values (calendar) | `calendar_heatmap` (Highcharts) |
| Two numeric variables | `bubblechart`, `contour_plot`, `hexagonal_binning` |
| Correlation matrix | `matrixplot` |
| Multi-dimensional | `parallelcoordinates` |
| Hierarchical data | `treemap`, `sunburst`, `circlepacking`, `dendrogram` |
| Flow/connections | `sankey_diagram`, `alluvial_diagram`, `chord_diagram`, `arc_diagram` |
| Grouped points | `convex_hull`, `voronoidiagram` |

## Color System

```python
from scomp_link.utils.colors import PRIMARY, LIGHT, MEDIUM, DARK, MAIN, MAIN_LIGHT, MAIN_DARK

# PRIMARY = 10 distinct colors for categorical data
# ["#6E37FA", "#32BBB9", "#FF9408", "#F40953", "#FA32A0",
#  "#B30095", "#FFD500", "#AAF564", "#50E6AA", "#2765F0"]
```

Use these when creating custom Plotly figures for visual consistency with scomp-link reports.
