# Utilities Module

## Overview

The utilities module provides supporting tools for HTML report generation, visualization, and helper functions used throughout the scomp-link package.

## Core Components

### ScompLinkHTMLReport

Programmatic HTML report builder with embedded Plotly visualizations:

```python
from scomp_link.utils.report_html import ScompLinkHTMLReport
import plotly.express as px

# Create report
report = ScompLinkHTMLReport(
    title='Analysis Report',
    main_color='#6E37FA',
    light_color='#9682FF',
    dark_color='#4614B4',
    font_family='Baloo 2',
    language='en'
)

# Build report with fluent API
report.open_section("Data Analysis")
report.add_title("Distribution Analysis")
report.add_text("This section shows key variable distributions.")

# Add Plotly graphs
fig = px.scatter(df, x='x', y='y', title='Scatter Plot')
report.add_graph_to_report(fig, 'Feature vs Target')

# Add DataFrames
report.add_dataframe(df.head(20), 'Sample Data')

report.close_section()

# Save self-contained HTML
report.save_html('analysis_report.html')
```

## Key Features

### 1. Fluent API (Builder Pattern)

Chain methods for intuitive report building:

```python
report = ScompLinkHTMLReport("Report")
report.open_section("Section 1") \
      .add_title("Title") \
      .add_text("Description") \
      .add_graph_to_report(fig, "Graph") \
      .close_section()
```

### 2. Interactive Plotly Graphs

Embed interactive visualizations:

```python
# Single plot
fig = px.line(df, x='date', y='value')
report.add_graph_to_report(fig, 'Time Series')

# Multiple plots with selection box
figures_dict = {
    'Plot 1': fig1,
    'Plot 2': fig2,
    'Plot 3': fig3
}
report.add_many_plots_with_selection_box_to_report(
    figures_dict,
    'Select Plot',
    labels=['Choose visualization']
)
```

### 3. DataFrame Rendering

Display data tables with export functionality:

```python
# Add table with CSV export
report.add_dataframe(
    df=results_df,
    title='Model Results',
    limit_max=2000  # Max rows to display
)
```

### 4. Collapsible Sections

Organize content in expandable sections:

```python
report.open_section("Detailed Analysis")
# Add content...
report.close_section()

# Sections are collapsible in final HTML
```

### 5. Responsive Design

Reports automatically adapt to screen size:
- Desktop: Full width with sidebars
- Tablet: Adjusted padding
- Mobile: Optimized layout

### 6. Custom Styling

Customize colors and branding:

```python
report = ScompLinkHTMLReport(
    title='Custom Report',
    main_color='#FF5733',      # Primary color
    light_color='#FFC300',     # Light accent
    dark_color='#C70039',      # Dark accent
    font_family='Arial',       # Font
    url_img_logo='logo.png',   # Optional logo
    url_background_header='bg.jpg'  # Optional header background
)
```

## Report Structure

### Standard Report Layout

```html
<!DOCTYPE html>
<html>
<head>
    <!-- Meta info, title, CSS, JS libraries -->
</head>
<body>
    <header>
        <!-- Title and branding -->
    </header>
    
    <div class="report">
        <!-- Collapsible sections -->
        <button class="collapsiblemygs">Section Title</button>
        <div class="content">
            <!-- Section content: text, graphs, tables -->
        </div>
    </div>
    
    <footer>
        <!-- About scomp-link -->
    </footer>
</body>
</html>
```

### Embedded Libraries

Reports include:
- **Plotly.js**: Interactive visualizations
- **jQuery**: DOM manipulation
- **Select2**: Enhanced dropdowns
- **Highcharts**: Additional charting (optional)

## Visualization Utilities

### Plotly Helper Functions

```python
from scomp_link.utils.plotly_utils import (
    histogram,
    multiple_histograms,
    barchart,
    linechart,
    area_chart
)

# Single histogram
fig = histogram(df['age'], 'Age Distribution', h=600)

# Multiple histograms by category
fig = multiple_histograms(
    df['value'],
    df['category'],
    category_name='Product Category',
    y_label='Sales'
)

# Bar chart with multiple series
fig = barchart(
    categories=['A', 'B', 'C'],
    metric_values_list=[[10, 20, 30], [15, 25, 35]],
    y_axis_titles=['Metric 1', 'Metric 2']
)

# Line chart
fig = linechart(
    date_list=['2024-01-01', '2024-01-02', '2024-01-03'],
    lines=[[10, 15, 20], [5, 10, 15]],
    y_labels=['Series 1', 'Series 2'],
    title_text='Time Series'
)
```

## Integration with Validation

The HTML report builder is automatically used by the Validator:

```python
from scomp_link.validation import Validator

validator = Validator(model)
validator.generate_validation_report(
    y_test, y_pred,
    task_type="regression",
    report_name="Validation_Report.html"
)
# Automatically creates comprehensive HTML report
```

## Report Features

### Automatic Features

1. **CSV Export**: All tables have download buttons
2. **Responsive Plots**: Auto-resize on window change
3. **Collapsible Sections**: Click to expand/collapse
4. **Hover Tooltips**: Interactive data exploration
5. **Zoom/Pan**: Full Plotly interactivity

### Customization Options

```python
# Custom section styling
report.open_section("Custom Section", ingore_multi_section=True)

# Custom plot with specific ID
report.single_plotly(fig, title="Plot", plotdivid="custom_id")

# Multiple plots with filters
report.select_plotly(
    figures_dict={
        ('Category A', 'Metric 1'): fig1,
        ('Category A', 'Metric 2'): fig2,
        ('Category B', 'Metric 1'): fig3
    },
    title="Filtered Plots",
    labels=['Category', 'Metric']
)
```

## Best Practices

1. **Section Organization**: Use sections to group related content
2. **Descriptive Titles**: Clear titles for graphs and tables
3. **Limit Table Size**: Use `limit_max` to avoid huge tables
4. **Color Consistency**: Use consistent color scheme
5. **Self-Contained**: Reports include all CSS/JS (no external dependencies)

## Advanced Usage

### Custom HTML Injection

```python
# Add custom HTML
report.html_report += """
<div class="custom-section">
    <h3>Custom Content</h3>
    <p>Your custom HTML here</p>
</div>
"""
```

### Dynamic Plot Selection

```python
# Create interactive plot selector
figures_by_category = {
    'Sales': sales_fig,
    'Revenue': revenue_fig,
    'Profit': profit_fig
}

report.add_many_plots_with_selection_box_to_report(
    figures_by_category,
    'Financial Metrics',
    labels=['Select Metric']
)
```

### Multi-Language Support

```python
# Italian report
report = ScompLinkHTMLReport(
    title='Rapporto di Analisi',
    language='it'
)

# Spanish report
report = ScompLinkHTMLReport(
    title='Informe de Análisis',
    language='es'
)
```

## Output Format

### File Structure

```
report.html (self-contained)
├── Embedded CSS
├── Embedded JavaScript
├── Plotly visualizations (JSON)
├── Data tables (HTML)
└── No external dependencies
```

### File Size

- Typical report: 500KB - 2MB
- With many plots: 2MB - 10MB
- All assets embedded (no external files needed)

## Dependencies

- plotly: Interactive visualizations
- pandas: DataFrame rendering
- json: Data serialization
- jwt: Optional token handling

## See Also

- [Validation Reports](../validation/README.md)
- [Visualization Examples](../../examples/)
- [Complete Pipeline](../README.md)
