# -*- coding: utf-8 -*-
"""
██████╗  █████╗ ██╗    ██╗ ██████╗ ██████╗  █████╗ ██████╗ ██╗  ██╗███████╗
██╔══██╗██╔══██╗██║    ██║██╔════╝ ██╔══██╗██╔══██╗██╔══██╗██║  ██║██╔════╝
██████╔╝███████║██║ █╗ ██║██║  ███╗██████╔╝███████║██████╔╝███████║███████╗
██╔══██╗██╔══██║██║███╗██║██║   ██║██╔══██╗██╔══██║██╔═══╝ ██╔══██║╚════██║
██║  ██║██║  ██║╚███╔███╔╝╚██████╔╝██║  ██║██║  ██║██║     ██║  ██║███████║
╚═╝  ╚═╝╚═╝  ╚═╝ ╚══╝╚══╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝╚══════╝

RAWGraphs-style SVG chart library for scomp-link.
Generates publication-quality SVG charts server-side.
"""

from .comparisons import barchart, barchartmultiset, barchartstacked, piechart, radarchart, voronoidiagram
from .distributions import beeswarm, boxplot, violinplot
from .time_series import bumpchart, gantt_chart, horizongraph, linechart, slopechart, streamgraph
from .correlations import bubblechart, contour_plot, convex_hull, hexagonal_binning, matrixplot, parallelcoordinates
from .hierarchies import circlepacking, circular_dendrogram, dendrogram, sunburst, treemap, voronoi_treemap
from .networks import alluvial_diagram, arc_diagram, chord_diagram, sankey_diagram

__all__ = [
    'barchart', 'barchartmultiset', 'barchartstacked', 'piechart', 'radarchart', 'voronoidiagram',
    'beeswarm', 'boxplot', 'violinplot',
    'bumpchart', 'gantt_chart', 'horizongraph', 'linechart', 'slopechart', 'streamgraph',
    'bubblechart', 'contour_plot', 'convex_hull', 'hexagonal_binning', 'matrixplot', 'parallelcoordinates',
    'circlepacking', 'circular_dendrogram', 'dendrogram', 'sunburst', 'treemap', 'voronoi_treemap',
    'alluvial_diagram', 'arc_diagram', 'chord_diagram', 'sankey_diagram',
]
