# -*- coding: utf-8 -*-
"""
██████╗  █████╗ ██╗    ██╗ ██████╗ ██████╗  █████╗ ██████╗ ██╗  ██╗███████╗
██╔══██╗██╔══██╗██║    ██║██╔════╝ ██╔══██╗██╔══██╗██╔══██╗██║  ██║██╔════╝
██████╔╝███████║██║ █╗ ██║██║  ███╗██████╔╝███████║██████╔╝███████║███████╗
██╔══██╗██╔══██║██║███╗██║██║   ██║██╔══██╗██╔══██║██╔═══╝ ██╔══██║╚════██║
██║  ██║██║  ██║╚███╔███╔╝╚██████╔╝██║  ██║██║  ██║██║     ██║  ██║███████║
╚═╝  ╚═╝╚═╝  ╚═╝ ╚══╝╚══╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝╚══════╝

RAWGraphs-style comparison chart functions for scomp-link.
Generates SVG charts server-side using matplotlib.
"""

import io
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Voronoi

from scomp_link.utils.colors import PRIMARY as COLORS


def _fig_to_svg(fig):
    """Convert a matplotlib figure to an embeddable SVG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format='svg', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    svg = buf.read().decode('utf-8')
    if svg.startswith('<?xml'):
        svg = svg[svg.index('<svg'):]
    return svg


def barchart(categories: list, values: list, title: str = '', width: int = 800, height: int = 500, colors: list = None) -> str:
    """Simple vertical bar chart. Each bar colored cycling through the palette."""
    palette = colors or COLORS
    bar_colors = [palette[i % len(palette)] for i in range(len(categories))]

    fig, ax = plt.subplots(figsize=(width / 100, height / 100))
    ax.bar(categories, values, color=bar_colors)
    if title:
        ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return _fig_to_svg(fig)


def barchartmultiset(categories: list, groups: dict, title: str = '', width: int = 800, height: int = 500, colors: list = None) -> str:
    """Grouped/multi-set bar chart with bars side by side per category."""
    palette = colors or COLORS
    n_groups = len(groups)
    x = np.arange(len(categories))
    bar_width = 0.8 / n_groups

    fig, ax = plt.subplots(figsize=(width / 100, height / 100))
    for i, (name, vals) in enumerate(groups.items()):
        offset = (i - n_groups / 2 + 0.5) * bar_width
        ax.bar(x + offset, vals, bar_width, label=name, color=palette[i % len(palette)])

    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    if title:
        ax.set_title(title)
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return _fig_to_svg(fig)


def barchartstacked(categories: list, groups: dict, title: str = '', width: int = 800, height: int = 500, colors: list = None) -> str:
    """Stacked bar chart."""
    palette = colors or COLORS
    fig, ax = plt.subplots(figsize=(width / 100, height / 100))

    bottom = np.zeros(len(categories))
    for i, (name, vals) in enumerate(groups.items()):
        ax.bar(categories, vals, bottom=bottom, label=name, color=palette[i % len(palette)])
        bottom += np.array(vals)

    if title:
        ax.set_title(title)
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return _fig_to_svg(fig)


def piechart(labels: list, values: list, title: str = '', width: int = 600, height: int = 600, colors: list = None) -> str:
    """Pie chart with labels."""
    palette = colors or COLORS
    pie_colors = [palette[i % len(palette)] for i in range(len(labels))]

    fig, ax = plt.subplots(figsize=(width / 100, height / 100))
    ax.pie(values, labels=labels, colors=pie_colors, autopct='%1.1f%%', startangle=90)
    if title:
        ax.set_title(title)
    return _fig_to_svg(fig)


def radarchart(categories: list, series: dict, title: str = '', width: int = 600, height: int = 600, colors: list = None) -> str:
    """Radar/spider chart using polar projection."""
    palette = colors or COLORS
    n = len(categories)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(width / 100, height / 100), subplot_kw={'polar': True})
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)

    for i, (name, vals) in enumerate(series.items()):
        data = vals + vals[:1]
        ax.plot(angles, data, label=name, color=palette[i % len(palette)], linewidth=2)
        ax.fill(angles, data, alpha=0.15, color=palette[i % len(palette)])

    if title:
        ax.set_title(title, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    return _fig_to_svg(fig)


def voronoidiagram(points: list, labels: list = None, title: str = '', width: int = 800, height: int = 600, colors: list = None) -> str:
    """Voronoi tessellation from 2D points, each region colored."""
    palette = colors or COLORS
    pts = np.array(points)
    vor = Voronoi(pts)

    fig, ax = plt.subplots(figsize=(width / 100, height / 100))

    # Color finite regions
    for i, region_idx in enumerate(vor.point_region):
        region = vor.regions[region_idx]
        if not region or -1 in region:
            continue
        polygon = [vor.vertices[v] for v in region]
        ax.fill(*zip(*polygon), alpha=0.6, color=palette[i % len(palette)])

    # Plot points
    ax.plot(pts[:, 0], pts[:, 1], 'ko', markersize=4)

    # Draw ridges
    for simplex in vor.ridge_vertices:
        if -1 not in simplex:
            ax.plot([vor.vertices[simplex[0], 0], vor.vertices[simplex[1], 0]],
                    [vor.vertices[simplex[0], 1], vor.vertices[simplex[1], 1]], 'k-', linewidth=0.5)

    if labels:
        for pt, label in zip(pts, labels):
            ax.annotate(label, pt, fontsize=8, ha='center', va='bottom')

    if title:
        ax.set_title(title)
    ax.set_xlim(pts[:, 0].min() - 0.1, pts[:, 0].max() + 0.1)
    ax.set_ylim(pts[:, 1].min() - 0.1, pts[:, 1].max() + 0.1)
    ax.set_aspect('equal')
    ax.axis('off')
    return _fig_to_svg(fig)


if __name__ == '__main__':
    import os
    import numpy as np
    from scomp_link.utils.report_html import ScompLinkHTMLReport

    os.makedirs('tmp', exist_ok=True)

    report = ScompLinkHTMLReport('RAWGraphs Comparisons Demo')

    # barchart
    svg = barchart(['Q1', 'Q2', 'Q3', 'Q4'], [120, 180, 150, 210], 'Quarterly Revenue')
    report.add_rawgraphs_to_report(svg, 'Bar Chart')

    # barchartmultiset
    svg = barchartmultiset(['Jan', 'Feb', 'Mar'], {'Product A': [30, 45, 28], 'Product B': [22, 38, 41]}, 'Sales by Product')
    report.add_rawgraphs_to_report(svg, 'Multi-set Bar Chart')

    # barchartstacked
    svg = barchartstacked(['2021', '2022', '2023'], {'Online': [100, 150, 200], 'Retail': [80, 90, 110]}, 'Revenue Channels')
    report.add_rawgraphs_to_report(svg, 'Stacked Bar Chart')

    # piechart
    svg = piechart(['Desktop', 'Mobile', 'Tablet'], [55, 35, 10], 'Device Share')
    report.add_rawgraphs_to_report(svg, 'Pie Chart')

    # radarchart
    svg = radarchart(['Speed', 'Power', 'Range', 'Durability', 'Accuracy'],
                     {'Player A': [8, 6, 7, 9, 5], 'Player B': [6, 9, 5, 7, 8]}, 'Player Stats')
    report.add_rawgraphs_to_report(svg, 'Radar Chart')

    # voronoidiagram
    np.random.seed(42)
    pts = np.random.rand(15, 2).tolist()
    svg = voronoidiagram(pts, title='Random Voronoi')
    report.add_rawgraphs_to_report(svg, 'Voronoi Diagram')

    report.save_html('tmp/demo_comparisons.html')
    print('Saved tmp/demo_comparisons.html')
