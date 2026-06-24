# -*- coding: utf-8 -*-
"""
██████╗  █████╗ ██╗    ██╗ ██████╗ ██████╗  █████╗ ██████╗ ██╗  ██╗███████╗
██╔══██╗██╔══██╗██║    ██║██╔════╝ ██╔══██╗██╔══██╗██╔══██╗██║  ██║██╔════╝
██████╔╝███████║██║ █╗ ██║██║  ███╗██████╔╝███████║██████╔╝███████║███████╗
██╔══██╗██╔══██║██║███╗██║██║   ██║██╔══██╗██╔══██║██╔═══╝ ██╔══██║╚════██║
██║  ██║██║  ██║╚███╔███╔╝╚██████╔╝██║  ██║██║  ██║██║     ██║  ██║███████║
╚═╝  ╚═╝╚═╝  ╚═╝ ╚══╝╚══╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝╚══════╝

RAWGraphs-style correlation chart functions for scomp-link.
Generates SVG charts server-side using matplotlib.
"""

import io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

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


def bubblechart(x: list, y: list, size: list, labels: list = None, title: str = '',
                width: int = 800, height: int = 600, colors: list = None) -> str:
    """Scatter plot with sized bubbles, colored by index cycling through palette."""
    colors = colors or COLORS
    fig, ax = plt.subplots(figsize=(width / 100, height / 100))
    s_arr = np.array(size)
    s_scaled = (s_arr / s_arr.max()) * 1000 if s_arr.max() > 0 else s_arr
    c = [colors[i % len(colors)] for i in range(len(x))]
    ax.scatter(x, y, s=s_scaled, c=c, alpha=0.6, edgecolors='white', linewidth=0.5)
    if labels:
        for i, lbl in enumerate(labels):
            ax.annotate(lbl, (x[i], y[i]), fontsize=7, ha='center')
    if title:
        ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.3)
    return _fig_to_svg(fig)


def contour_plot(x: list, y: list, title: str = '', width: int = 800, height: int = 600,
                 colors: list = None, levels: int = 10) -> str:
    """2D kernel density contour plot using scipy gaussian_kde."""
    from scipy.stats import gaussian_kde
    colors = colors or COLORS
    fig, ax = plt.subplots(figsize=(width / 100, height / 100))
    xy = np.vstack([x, y])
    kde = gaussian_kde(xy)
    xmin, xmax = min(x), max(x)
    ymin, ymax = min(y), max(y)
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    cmap = LinearSegmentedColormap.from_list('custom', [colors[0], colors[1], colors[2]])
    ax.contourf(xx, yy, zz, levels=levels, cmap=cmap, alpha=0.8)
    ax.scatter(x, y, s=5, color=colors[3], alpha=0.4)
    if title:
        ax.set_title(title, fontweight='bold')
    return _fig_to_svg(fig)


def convex_hull(x: list, y: list, groups: list, title: str = '', width: int = 800,
                height: int = 600, colors: list = None) -> str:
    """Scatter plot with convex hull drawn around each group."""
    from scipy.spatial import ConvexHull
    colors = colors or COLORS
    fig, ax = plt.subplots(figsize=(width / 100, height / 100))
    unique_groups = list(dict.fromkeys(groups))
    for gi, grp in enumerate(unique_groups):
        c = colors[gi % len(colors)]
        idx = [i for i, g in enumerate(groups) if g == grp]
        gx = np.array([x[i] for i in idx])
        gy = np.array([y[i] for i in idx])
        ax.scatter(gx, gy, color=c, label=grp, alpha=0.7, edgecolors='white', linewidth=0.5)
        if len(idx) >= 3:
            points = np.column_stack([gx, gy])
            hull = ConvexHull(points)
            hull_pts = np.append(hull.vertices, hull.vertices[0])
            ax.fill(points[hull_pts, 0], points[hull_pts, 1], alpha=0.15, color=c)
            ax.plot(points[hull_pts, 0], points[hull_pts, 1], color=c, linewidth=1.5, alpha=0.6)
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.3)
    return _fig_to_svg(fig)


def hexagonal_binning(x: list, y: list, title: str = '', width: int = 800, height: int = 600,
                      colors: list = None, gridsize: int = 15) -> str:
    """Hexagonal bin density plot."""
    colors = colors or COLORS
    fig, ax = plt.subplots(figsize=(width / 100, height / 100))
    cmap = LinearSegmentedColormap.from_list('custom', [colors[0], colors[1], colors[2]])
    hb = ax.hexbin(x, y, gridsize=gridsize, cmap=cmap, mincnt=1)
    fig.colorbar(hb, ax=ax, shrink=0.8)
    if title:
        ax.set_title(title, fontweight='bold')
    return _fig_to_svg(fig)


def matrixplot(matrix: list, row_labels: list = None, col_labels: list = None, title: str = '',
               width: int = 700, height: int = 700, colors: list = None) -> str:
    """Heatmap matrix with colorbar."""
    colors = colors or COLORS
    fig, ax = plt.subplots(figsize=(width / 100, height / 100))
    arr = np.array(matrix)
    cmap = LinearSegmentedColormap.from_list('custom', [colors[0], colors[6], colors[2]])
    im = ax.imshow(arr, cmap=cmap, aspect='auto')
    fig.colorbar(im, ax=ax, shrink=0.8)
    if row_labels:
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=8)
    if col_labels:
        ax.set_xticks(range(len(col_labels)))
        ax.set_xticklabels(col_labels, fontsize=8, rotation=45, ha='right')
    if title:
        ax.set_title(title, fontweight='bold')
    return _fig_to_svg(fig)


def parallelcoordinates(data: dict, class_column: str = None, title: str = '',
                        width: int = 800, height: int = 500, colors: list = None) -> str:
    """Parallel coordinates plot. Each row is a line connecting normalized values across axes."""
    colors = colors or COLORS
    fig, ax = plt.subplots(figsize=(width / 100, height / 100))
    cols = [k for k in data if k != class_column]
    n_rows = len(data[cols[0]])
    # Normalize each column to [0, 1]
    normed = {}
    for col in cols:
        vals = np.array(data[col], dtype=float)
        vmin, vmax = vals.min(), vals.max()
        normed[col] = (vals - vmin) / (vmax - vmin) if vmax != vmin else np.zeros_like(vals)
    # Determine line colors
    if class_column and class_column in data:
        classes = data[class_column]
        unique_classes = list(dict.fromkeys(classes))
        color_map = {c: colors[i % len(colors)] for i, c in enumerate(unique_classes)}
        line_colors = [color_map[c] for c in classes]
    else:
        line_colors = [colors[i % len(colors)] for i in range(n_rows)]
    # Draw lines
    x_pos = range(len(cols))
    for i in range(n_rows):
        y_vals = [normed[col][i] for col in cols]
        ax.plot(x_pos, y_vals, color=line_colors[i], alpha=0.5, linewidth=0.8)
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(cols, fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel('Normalized value', fontsize=8)
    if class_column and class_column in data:
        from matplotlib.lines import Line2D
        unique_classes = list(dict.fromkeys(data[class_column]))
        legend_elements = [Line2D([0], [0], color=colors[i % len(colors)], label=c)
                           for i, c in enumerate(unique_classes)]
        ax.legend(handles=legend_elements, fontsize=8, loc='upper right')
    if title:
        ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    return _fig_to_svg(fig)


if __name__ == '__main__':
    import numpy as np
    from scomp_link.utils.report_html import ScompLinkHTMLReport
    import os
    os.makedirs('tmp', exist_ok=True)

    report = ScompLinkHTMLReport('RAWGraphs Correlations Demo')
    np.random.seed(42)

    # bubblechart
    x = np.random.rand(20).tolist()
    y = np.random.rand(20).tolist()
    s = (np.random.rand(20) * 100).tolist()
    svg = bubblechart(x, y, s, title='Market Analysis')
    report.add_rawgraphs_to_report(svg, 'Bubble Chart')

    # contour_plot
    x = np.random.normal(0, 1, 200).tolist()
    y = np.random.normal(0, 1, 200).tolist()
    svg = contour_plot(x, y, 'Density Estimation')
    report.add_rawgraphs_to_report(svg, 'Contour Plot')

    # convex_hull
    x = np.concatenate([np.random.normal(0, 1, 30), np.random.normal(3, 1, 30)]).tolist()
    y = np.concatenate([np.random.normal(0, 1, 30), np.random.normal(3, 1, 30)]).tolist()
    groups = ['A']*30 + ['B']*30
    svg = convex_hull(x, y, groups, 'Cluster Boundaries')
    report.add_rawgraphs_to_report(svg, 'Convex Hull')

    # hexagonal_binning
    x = np.random.normal(0, 2, 500).tolist()
    y = np.random.normal(0, 2, 500).tolist()
    svg = hexagonal_binning(x, y, '2D Density')
    report.add_rawgraphs_to_report(svg, 'Hexagonal Binning')

    # matrixplot
    matrix = np.random.rand(6, 6).tolist()
    svg = matrixplot(matrix, ['A', 'B', 'C', 'D', 'E', 'F'], ['X', 'Y', 'Z', 'W', 'V', 'U'], 'Correlation Matrix')
    report.add_rawgraphs_to_report(svg, 'Matrix Plot')

    # parallelcoordinates
    data = {'Speed': np.random.rand(30).tolist(), 'Power': np.random.rand(30).tolist(),
            'Efficiency': np.random.rand(30).tolist(), 'Cost': np.random.rand(30).tolist(),
            'Type': (['A']*15 + ['B']*15)}
    svg = parallelcoordinates(data, 'Type', 'Performance Comparison')
    report.add_rawgraphs_to_report(svg, 'Parallel Coordinates')

    report.save_html('tmp/demo_correlations.html')
    print('Saved tmp/demo_correlations.html')
