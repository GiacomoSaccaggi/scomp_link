# -*- coding: utf-8 -*-
"""
Example 19: RAWGraphs SVG Charts
=================================
Demonstrates generating server-side SVG charts using the rawgraphs module
and embedding them into an HTML report.
"""
import os
import numpy as np
from scipy.cluster.hierarchy import linkage

from scomp_link.utils.report_html import ScompLinkHTMLReport
from scomp_link.utils.rawgraphs import (
    barchart, barchartmultiset, barchartstacked, piechart, radarchart, voronoidiagram,
    beeswarm, boxplot, violinplot,
    bumpchart, gantt_chart, horizongraph, linechart, slopechart, streamgraph,
    bubblechart, contour_plot, convex_hull, hexagonal_binning, matrixplot, parallelcoordinates,
    circlepacking, circular_dendrogram, dendrogram, sunburst, treemap, voronoi_treemap,
    alluvial_diagram, arc_diagram, chord_diagram, sankey_diagram,
)

os.makedirs('staging', exist_ok=True)
np.random.seed(42)

report = ScompLinkHTMLReport('RAWGraphs Charts — Full Demo')

# ── Comparisons ──
report.open_section('Comparisons')
report.add_rawgraphs_to_report(
    barchart(['Q1', 'Q2', 'Q3', 'Q4'], [120, 180, 150, 210], 'Quarterly Revenue'),
    'Bar Chart')
report.add_rawgraphs_to_report(
    barchartmultiset(['Jan', 'Feb', 'Mar'],
                     {'Product A': [30, 45, 28], 'Product B': [22, 38, 41]}, 'Sales'),
    'Multi-set Bar Chart')
report.add_rawgraphs_to_report(
    barchartstacked(['2021', '2022', '2023'],
                    {'Online': [100, 150, 200], 'Retail': [80, 90, 110]}, 'Revenue'),
    'Stacked Bar Chart')
report.add_rawgraphs_to_report(
    piechart(['Desktop', 'Mobile', 'Tablet'], [55, 35, 10], 'Device Share'),
    'Pie Chart')
report.add_rawgraphs_to_report(
    radarchart(['Speed', 'Power', 'Range', 'Durability', 'Accuracy'],
               {'Player A': [8, 6, 7, 9, 5], 'Player B': [6, 9, 5, 7, 8]}, 'Stats'),
    'Radar Chart')
report.add_rawgraphs_to_report(
    voronoidiagram(np.random.rand(12, 2).tolist(), title='Voronoi'),
    'Voronoi Diagram')
report.close_section()

# ── Distributions ──
report.open_section('Distributions')
data = np.concatenate([np.random.normal(5, 1, 40), np.random.normal(8, 1.5, 40)])
groups = ['Control'] * 40 + ['Treatment'] * 40
report.add_rawgraphs_to_report(beeswarm(data.tolist(), groups, 'Scores'), 'Beeswarm')
data_box = [np.random.normal(10, 2, 80).tolist(), np.random.normal(14, 3, 80).tolist()]
report.add_rawgraphs_to_report(boxplot(data_box, ['A', 'B'], 'Groups'), 'Box Plot')
report.add_rawgraphs_to_report(violinplot(data_box, ['A', 'B'], 'Density'), 'Violin Plot')
report.close_section()

# ── Time Series ──
report.open_section('Time Series')
report.add_rawgraphs_to_report(
    bumpchart({'Apple': [1, 2, 1], 'Google': [2, 1, 2], 'MS': [3, 3, 3]},
              ['2021', '2022', '2023'], 'Rank'), 'Bump Chart')
tasks = [{'name': 'Design', 'start': 0, 'end': 3, 'group': 'P1'},
         {'name': 'Build', 'start': 2, 'end': 7, 'group': 'P1'},
         {'name': 'Test', 'start': 6, 'end': 9, 'group': 'P2'}]
report.add_rawgraphs_to_report(gantt_chart(tasks, 'Timeline'), 'Gantt Chart')
x = list(range(30))
report.add_rawgraphs_to_report(
    horizongraph({'Temp': (np.sin(np.linspace(0, 4 * np.pi, 30)) * 8).tolist()}, x, 'Sensor'),
    'Horizon Graph')
report.add_rawgraphs_to_report(
    linechart({'Revenue': [10, 15, 13, 18, 22], 'Costs': [8, 9, 11, 12, 14]},
              ['Jan', 'Feb', 'Mar', 'Apr', 'May'], 'Trend'), 'Line Chart')
report.add_rawgraphs_to_report(
    slopechart({'A': [45, 62], 'B': [70, 55]}, ['2022', '2023'], 'Change'),
    'Slope Chart')
report.add_rawgraphs_to_report(
    streamgraph({'Rock': np.random.randint(5, 20, 15).tolist(),
                 'Pop': np.random.randint(3, 15, 15).tolist()}, list(range(15)), 'Music'),
    'Streamgraph')
report.close_section()

# ── Correlations ──
report.open_section('Correlations')
report.add_rawgraphs_to_report(
    bubblechart(np.random.rand(15).tolist(), np.random.rand(15).tolist(),
                (np.random.rand(15) * 80).tolist(), title='Bubbles'), 'Bubble Chart')
report.add_rawgraphs_to_report(
    contour_plot(np.random.normal(0, 1, 150).tolist(),
                 np.random.normal(0, 1, 150).tolist(), 'KDE'), 'Contour Plot')
x = np.concatenate([np.random.normal(0, 1, 25), np.random.normal(3, 1, 25)]).tolist()
y = np.concatenate([np.random.normal(0, 1, 25), np.random.normal(3, 1, 25)]).tolist()
report.add_rawgraphs_to_report(
    convex_hull(x, y, ['A'] * 25 + ['B'] * 25, 'Clusters'), 'Convex Hull')
report.add_rawgraphs_to_report(
    hexagonal_binning(np.random.normal(0, 2, 300).tolist(),
                      np.random.normal(0, 2, 300).tolist(), 'Density'), 'Hex Binning')
report.add_rawgraphs_to_report(
    matrixplot(np.random.rand(5, 5).tolist(),
               ['A', 'B', 'C', 'D', 'E'], ['V', 'W', 'X', 'Y', 'Z'], 'Heatmap'),
    'Matrix Plot')
data = {'Speed': np.random.rand(20).tolist(), 'Power': np.random.rand(20).tolist(),
        'Range': np.random.rand(20).tolist(), 'Type': ['X'] * 10 + ['Y'] * 10}
report.add_rawgraphs_to_report(parallelcoordinates(data, 'Type', 'Perf'), 'Parallel Coords')
report.close_section()

# ── Hierarchies ──
report.open_section('Hierarchies')
hier = {'name': 'World', 'children': [
    {'name': 'EU', 'children': [{'name': 'FR', 'value': 67}, {'name': 'DE', 'value': 83}]},
    {'name': 'Asia', 'children': [{'name': 'CN', 'value': 140}, {'name': 'JP', 'value': 126}]},
]}
report.add_rawgraphs_to_report(circlepacking(hier, 'Population'), 'Circle Packing')
Z = linkage(np.random.rand(7, 3), method='ward')
report.add_rawgraphs_to_report(dendrogram(Z, list('ABCDEFG'), 'Clusters'), 'Dendrogram')
report.add_rawgraphs_to_report(circular_dendrogram(Z, list('ABCDEFG'), 'Circular'), 'Circular Dendro')
report.add_rawgraphs_to_report(sunburst(hier, 'Sunburst'), 'Sunburst')
tree = {'name': 'Budget', 'children': [
    {'name': 'Eng', 'value': 40}, {'name': 'Mkt', 'value': 25},
    {'name': 'Sales', 'value': 20}, {'name': 'HR', 'value': 15}]}
report.add_rawgraphs_to_report(treemap(tree, 'Departments'), 'Treemap')
report.add_rawgraphs_to_report(voronoi_treemap(tree, 'Voronoi'), 'Voronoi Treemap')
report.close_section()

# ── Networks ──
report.open_section('Networks')
flows = [{'source': 'Home', 'target': 'Products', 'value': 40},
         {'source': 'Home', 'target': 'Blog', 'value': 15},
         {'source': 'Products', 'target': 'Cart', 'value': 30}]
report.add_rawgraphs_to_report(alluvial_diagram(flows, 'User Flow'), 'Alluvial Diagram')
report.add_rawgraphs_to_report(
    arc_diagram(['A', 'B', 'C', 'D'],
                [{'source': 0, 'target': 2, 'value': 5}, {'source': 1, 'target': 3, 'value': 3}],
                'Network'), 'Arc Diagram')
report.add_rawgraphs_to_report(
    chord_diagram([[0, 5, 3], [5, 0, 4], [3, 4, 0]], ['X', 'Y', 'Z'], 'Flows'),
    'Chord Diagram')
nodes = [{'name': 'Solar', 'x': 0}, {'name': 'Wind', 'x': 0},
         {'name': 'Grid', 'x': 1}, {'name': 'Home', 'x': 2}]
links = [{'source': 0, 'target': 2, 'value': 40}, {'source': 1, 'target': 2, 'value': 30},
         {'source': 2, 'target': 3, 'value': 60}]
report.add_rawgraphs_to_report(sankey_diagram(nodes, links, 'Energy'), 'Sankey Diagram')
report.close_section()

report.save_html('staging/example_19_rawgraphs.html')
print('✅ Saved staging/example_19_rawgraphs.html — 31 charts rendered')
