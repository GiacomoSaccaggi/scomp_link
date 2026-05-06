# -*- coding: utf-8 -*-
"""
Example 17: HTML Report Generation & PDF Export

Demonstrates all features of ScompLinkHTMLReport:
- Titles and text
- DataFrames
- Collapsible sections
- Single Plotly graphs
- Combobox graph selection (single and multiple filters)
- Matplotlib graphs (base64)
- Local images (base64)
- Plotly utility charts (histogram, barchart, linechart, area_chart)
- HTML export
- PDF export (requires: pip install playwright && playwright install chromium)
"""

from scomp_link.utils.report_html import ScompLinkHTMLReport
from scomp_link.utils.plotly_utils import (
    multiple_histograms, histogram, barchart, area_chart, linechart
)
import pandas as pd
import numpy as np
import plotly.express as px

# ── Initialize report ──
demo_report = ScompLinkHTMLReport('My first REPORT')

# ── Add Title ──
demo_report.add_title('Lorem Ipsum')

# ── Add paragraph ──
demo_report.add_text('''Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor
invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et
ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet. Lorem ipsum dolor sit
amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor invidunt ut labore et dolore magna aliquyam erat,
sed diam voluptua. At vero eos et accusam et justo duo dolores et ea rebum. Stet clita kasd gubergren,
no sea takimata sanctus est Lorem ipsum dolor sit amet.''')

# ── Add Pandas DF ──
df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=['col1', 'col2', 'col3'])
demo_report.add_dataframe(df, 'demo_df')

# ── Sections ──
demo_report.open_section('My first section')
demo_report.add_title('1. Lorem Ipsum')
demo_report.add_text('''Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor
invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et
ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet.''')
demo_report.close_section()

demo_report.open_section('My second section')
demo_report.add_title('2. Lorem Ipsum')
demo_report.add_text('''Lorem ipsum dolor sit amet, consetetur sadipscing elitr, sed diam nonumy eirmod tempor
invidunt ut labore et dolore magna aliquyam erat, sed diam voluptua. At vero eos et accusam et justo duo dolores et
ea rebum. Stet clita kasd gubergren, no sea takimata sanctus est Lorem ipsum dolor sit amet.''')
demo_report.close_section()

# ── Add single graph ──
fig = px.scatter(x=range(10), y=range(10))
demo_report.add_graph_to_report(fig, 'My first Graph')

# ── Add graphs with combobox selections ──
fig1 = px.scatter(x=range(10), y=range(10))
fig2 = px.scatter(x=range(20), y=range(20))
figures_dict = {
    'UAT': fig1,
    'PROD': fig2
}
demo_report.add_many_plots_with_selection_box_to_report(figures_dict, 'My second Graph')

# ── Add graphs with multiple combobox selections ──
fig1a = px.scatter(x=range(10), y=range(10))
fig2a = px.scatter(x=range(20), y=range(20))
fig1b = px.scatter(x=range(10), y=range(10))
fig1b.update_traces(marker=dict(color='red'))
fig2b = px.scatter(x=range(20), y=range(20))
fig2b.update_traces(marker=dict(color='red'))
figures_dict = {
    ('This is the first', 'Blue'): fig1a,
    ('This is the second', 'Blue'): fig2a,
    ('This is the first', 'Red'): fig1b,
    ('This is the second', 'Red'): fig2b
}
demo_report.add_many_plots_with_selection_box_to_report(figures_dict, 'My third Graph')

# ── Add graphs with multiple combobox selections with labels name ──
fig1a = px.scatter(x=range(10), y=range(10))
fig2a = px.scatter(x=range(20), y=range(20))
fig1b = px.scatter(x=range(10), y=range(10))
fig1b.update_traces(marker=dict(color='red'))
fig2b = px.scatter(x=range(20), y=range(20))
fig2b.update_traces(marker=dict(color='red'))
figures_dict = {
    ('This is the first', 'Blue'): fig1a,
    ('This is the second', 'Blue'): fig2a,
    ('This is the first', 'Red'): fig1b,
    ('This is the second', 'Red'): fig2b
}
demo_report.add_many_plots_with_selection_box_to_report(
    figures_dict, 'My fourth Graph', labels=['Number', 'Color']
)

# ── Multiple Histogram ──
x1 = np.random.normal(85, 3, 1000)
x2 = [['O', 'CO2', 'H'][i] for i in np.random.randint(0, 3, 1000)]
fig = multiple_histograms(x1, x2, 'Comparison Gas')
demo_report.add_graph_to_report(fig, 'multiple_histograms')

# ── Single Histogram ──
x1 = np.random.normal(45, 3, 1000)
fig = histogram(x1, 'Panelist age')
demo_report.add_graph_to_report(fig, 'histogram')

# ── Barchart ──
categorie = ['C', 'A', 'E', 'B', 'D']
valori = [50, 80, 30, 70, 60]
obiettivo = [60, 75, 40, 85, 55]
ordine_personalizzato = ['A', 'B', 'C', 'D', 'E']

fig = barchart(
    categories=categorie,
    metric_values_list=[valori],
    metric_values_line_list=[obiettivo],
    categorysorted=ordine_personalizzato,
    order='asc',
    y_axis_titles='Valore (%)',
    y_line_axis_titles='Obiettivo'
)
demo_report.add_graph_to_report(fig, 'barchart')

# ── Area chart ──
date = ['2024-01-01', '2024-02-01', '2024-03-01', '2024-04-01', '2024-05-01']
active_users = [[5000 * (5 - i), 6200 * (5 - i), 5800 * (5 - i), 6500 * (5 - i), 7000 * (5 - i)] for i in range(2)]

fig = area_chart(
    date_list=date,
    lines=active_users,
    title_text='Trend Active Users',
    y_labels=['Number of Views 1', 'Number of Views 2'],
    format_date='%Y-%m-%d'
)
demo_report.add_graph_to_report(fig, 'area_chart')

# ── Linechart ──
date = ['2024-01-01', '2024-02-01', '2024-03-01', '2024-04-01', '2024-05-01']
views = [[5000 * (5 - i), 6200 * (5 - i), 5800 * (5 - i), 6500 * (5 - i), 7000 * (5 - i)] for i in range(2)]

fig = linechart(
    date_list=date,
    lines=views,
    title_text='Trend Views',
    y_labels=['Number of Views 1', 'Number of Views 2'],
    format_date='%Y-%m-%d'
)
demo_report.add_graph_to_report(fig, 'linechart')

# ── Matplotlib graph (base64) ──
import matplotlib.pyplot as plt

mpl_fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(np.linspace(0, 10, 100), np.sin(np.linspace(0, 10, 100)), label='sin(x)')
ax.set_title('Matplotlib Example')
ax.legend()
demo_report.add_matplotlib_graph_to_report(mpl_fig, 'Matplotlib Sin Wave')
plt.close(mpl_fig)

# ── Save HTML report ──
demo_report.save_html('demo_report.html')

# ── Save PDF report (requires playwright) ──
try:
    demo_report.save_pdf('demo_report.pdf')
except Exception as e:
    print(f"PDF export skipped: {e}")
    print("To enable PDF export run: pip install playwright && playwright install chromium")
