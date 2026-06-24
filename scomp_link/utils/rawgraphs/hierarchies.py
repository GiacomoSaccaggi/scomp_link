# -*- coding: utf-8 -*-
"""
██████╗  █████╗ ██╗    ██╗ ██████╗ ██████╗  █████╗ ██████╗ ██╗  ██╗███████╗
██╔══██╗██╔══██╗██║    ██║██╔════╝ ██╔══██╗██╔══██╗██╔══██╗██║  ██║██╔════╝
██████╔╝███████║██║ █╗ ██║██║  ███╗██████╔╝███████║██████╔╝███████║███████╗
██╔══██╗██╔══██║██║███╗██║██║   ██║██╔══██╗██╔══██║██╔═══╝ ██╔══██║╚════██║
██║  ██║██║  ██║╚███╔███╔╝╚██████╔╝██║  ██║██║  ██║██║     ██║  ██║███████║
╚═╝  ╚═╝╚═╝  ╚═╝ ╚══╝╚══╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝╚══════╝

RAWGraphs-style hierarchy chart functions for scomp-link.
Generates SVG charts server-side using matplotlib and scipy.
"""

import io
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Wedge
from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram
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


def _get_colors(colors):
    """Return color palette, defaulting to COLORS."""
    return colors if colors is not None else COLORS


def _flatten_leaves(node):
    """Recursively flatten hierarchy to list of (name, value) tuples."""
    if 'children' in node and node['children']:
        leaves = []
        for child in node['children']:
            leaves.extend(_flatten_leaves(child))
        return leaves
    return [(node.get('name', ''), node.get('value', 1))]


def _get_node_value(node):
    """Get total value of a node (sum of children or own value)."""
    if 'children' in node and node['children']:
        return sum(_get_node_value(c) for c in node['children'])
    return node.get('value', 1)


def _squarify(values, x, y, w, h):
    """Squarified treemap layout. Returns list of (x, y, w, h) rects."""
    if not values:
        return []
    if len(values) == 1:
        return [(x, y, w, h)]
    total = sum(values)
    if total == 0:
        return [(x, y, w, h)] * len(values)
    rects = []
    if w >= h:
        mid = 0
        s = 0
        for i, v in enumerate(values):
            s += v
            if s >= total / 2:
                mid = i + 1
                break
        left_vals = values[:mid]
        right_vals = values[mid:]
        left_w = w * sum(left_vals) / total
        rects.extend(_squarify(left_vals, x, y, left_w, h))
        rects.extend(_squarify(right_vals, x + left_w, y, w - left_w, h))
    else:
        mid = 0
        s = 0
        for i, v in enumerate(values):
            s += v
            if s >= total / 2:
                mid = i + 1
                break
        top_vals = values[:mid]
        bot_vals = values[mid:]
        top_h = h * sum(top_vals) / total
        rects.extend(_squarify(top_vals, x, y, w, top_h))
        rects.extend(_squarify(bot_vals, x, y + top_h, w, h - top_h))
    return rects


def circlepacking(data: dict, title: str = '', width: int = 700, height: int = 700, colors: list = None) -> str:
    """Circle packing visualization for hierarchical data."""
    palette = _get_colors(colors)
    leaves = _flatten_leaves(data)
    names = [l[0] for l in leaves]
    values = [l[1] for l in leaves]

    # Radius proportional to sqrt(value)
    radii = [math.sqrt(v) for v in values]
    max_r = max(radii) if radii else 1

    fig, ax = plt.subplots(1, 1, figsize=(width / 100, height / 100))
    ax.set_aspect('equal')
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')

    n = len(leaves)
    if n == 0:
        return _fig_to_svg(fig)

    # Arrange circles by angle around center
    cx, cy = 0.5, 0.5
    container_r = 0.45

    if n == 1:
        positions = [(cx, cy)]
        scale = container_r * 0.8
        radii_scaled = [scale]
    else:
        # Normalize radii to fit inside container
        total_r = sum(radii)
        radii_scaled = [r / total_r * container_r * 0.7 for r in radii]

        # Position using angular arrangement with distance from center based on index
        positions = []
        angle_step = 2 * math.pi / n
        for i in range(n):
            angle = i * angle_step
            dist = container_r * 0.55
            px = cx + dist * math.cos(angle)
            py = cy + dist * math.sin(angle)
            positions.append((px, py))

    # Draw container circle
    container = plt.Circle((cx, cy), container_r, fill=False, edgecolor='#333', linewidth=1.5, linestyle='--')
    ax.add_patch(container)

    # Draw leaf circles
    for i, (pos, r, name) in enumerate(zip(positions, radii_scaled, names)):
        color = palette[i % len(palette)]
        circle = plt.Circle(pos, r, facecolor=color, edgecolor='white', linewidth=1, alpha=0.8)
        ax.add_patch(circle)
        if r > 0.02:
            ax.text(pos[0], pos[1], name, ha='center', va='center', fontsize=7, color='white', fontweight='bold')

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    return _fig_to_svg(fig)


def circular_dendrogram(linkage_matrix, labels: list = None, title: str = '', width: int = 700, height: int = 700, colors: list = None) -> str:
    """Dendrogram in polar/circular layout."""
    palette = _get_colors(colors)

    # Get dendrogram data without plotting
    fig_tmp, ax_tmp = plt.subplots()
    ddata = scipy_dendrogram(linkage_matrix, labels=labels, no_plot=True)
    plt.close(fig_tmp)

    fig, ax = plt.subplots(1, 1, figsize=(width / 100, height / 100), subplot_kw={'projection': 'polar'})
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')

    n_leaves = len(ddata['leaves'])
    # Map x coordinates to angles
    x_coords = sorted(set(x for xs in ddata['icoord'] for x in xs))
    x_min, x_max = min(x_coords), max(x_coords)

    def x_to_angle(x):
        if x_max == x_min:
            return 0
        return 2 * math.pi * (x - x_min) / (x_max - x_min)

    # Max height for radial scaling
    y_coords = [y for ys in ddata['dcoord'] for y in ys]
    y_max = max(y_coords) if y_coords else 1

    def y_to_radius(y):
        return 0.3 + 0.6 * (1 - y / y_max) if y_max > 0 else 0.9

    # Draw links
    for i, (xs, ys) in enumerate(zip(ddata['icoord'], ddata['dcoord'])):
        color = palette[i % len(palette)]
        angles = [x_to_angle(x) for x in xs]
        radii = [y_to_radius(y) for y in ys]
        # Draw U-shape: vertical-horizontal-vertical in polar
        ax.plot([angles[0], angles[0]], [radii[0], radii[1]], color=color, linewidth=1.5)
        # Horizontal arc
        arc_angles = np.linspace(angles[1], angles[2], 30)
        arc_radii = [radii[1]] * 30
        ax.plot(arc_angles, arc_radii, color=color, linewidth=1.5)
        ax.plot([angles[3], angles[3]], [radii[2], radii[3]], color=color, linewidth=1.5)

    # Draw labels at leaves
    if labels is None:
        labels = [str(i) for i in range(n_leaves)]
    leaf_xs = sorted(set(ddata['icoord'][i][j] for i in range(len(ddata['icoord'])) for j in [0, 3]
                         if ddata['dcoord'][i][j] == 0))
    for lx, label in zip(leaf_xs, [labels[i] for i in ddata['leaves']]):
        angle = x_to_angle(lx)
        ax.text(angle, 1.0, label, ha='center', va='center', fontsize=8, fontweight='bold')

    ax.set_ylim(0, 1.1)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.grid(False)
    return _fig_to_svg(fig)


def dendrogram(linkage_matrix, labels: list = None, title: str = '', width: int = 800, height: int = 500, colors: list = None) -> str:
    """Standard linear dendrogram."""
    palette = _get_colors(colors)

    fig, ax = plt.subplots(1, 1, figsize=(width / 100, height / 100))

    # Use scipy dendrogram with color threshold to get colored clusters
    ddata = scipy_dendrogram(
        linkage_matrix,
        labels=labels,
        ax=ax,
        above_threshold_color='#333',
        leaf_font_size=9,
    )

    # Recolor the dendrogram links with our palette
    for i, line in enumerate(ax.get_lines()):
        line.set_color(palette[i % len(palette)])
        line.set_linewidth(1.5)

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_ylabel('Distance', fontsize=10)
    return _fig_to_svg(fig)


def sunburst(data: dict, title: str = '', width: int = 700, height: int = 700, colors: list = None) -> str:
    """Sunburst chart (nested rings)."""
    palette = _get_colors(colors)

    # Build levels: each level is a list of (name, value, color_index)
    def build_levels(node, depth=0, color_idx=0, levels=None):
        if levels is None:
            levels = []
        if 'children' in node and node['children']:
            while len(levels) <= depth:
                levels.append([])
            for i, child in enumerate(node['children']):
                cidx = (color_idx + i) % len(palette)
                val = _get_node_value(child)
                levels[depth].append((child.get('name', ''), val, cidx))
                build_levels(child, depth + 1, cidx, levels)
        return levels

    levels = build_levels(data)
    if not levels:
        fig, ax = plt.subplots(1, 1, figsize=(width / 100, height / 100))
        ax.axis('off')
        return _fig_to_svg(fig)

    fig, ax = plt.subplots(1, 1, figsize=(width / 100, height / 100))
    ax.set_aspect('equal')
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')

    n_levels = len(levels)
    ring_width = 0.3 / n_levels if n_levels > 0 else 0.3

    for depth, level in enumerate(levels):
        inner_r = 0.15 + depth * ring_width
        outer_r = inner_r + ring_width * 0.9
        total = sum(item[1] for item in level)
        if total == 0:
            continue

        start_angle = 0
        for name, value, cidx in level:
            sweep = 360 * value / total
            theta1 = start_angle
            theta2 = start_angle + sweep

            wedge = Wedge(
                (0.5, 0.5), outer_r, theta1, theta2,
                width=outer_r - inner_r,
                facecolor=palette[cidx],
                edgecolor='white',
                linewidth=1,
                alpha=0.85
            )
            ax.add_patch(wedge)

            # Label in the middle of the wedge
            if sweep > 15:
                mid_angle = math.radians((theta1 + theta2) / 2)
                mid_r = (inner_r + outer_r) / 2
                lx = 0.5 + mid_r * math.cos(mid_angle)
                ly = 0.5 + mid_r * math.sin(mid_angle)
                ax.text(lx, ly, name, ha='center', va='center', fontsize=6, color='white', fontweight='bold')

            start_angle += sweep

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return _fig_to_svg(fig)


def treemap(data: dict, title: str = '', width: int = 800, height: int = 600, colors: list = None) -> str:
    """Treemap with area-proportional rectangles."""
    palette = _get_colors(colors)

    children = data.get('children', [])
    if not children:
        fig, ax = plt.subplots(1, 1, figsize=(width / 100, height / 100))
        ax.axis('off')
        return _fig_to_svg(fig)

    names = [c.get('name', '') for c in children]
    values = [_get_node_value(c) for c in children]

    # Sort descending for better layout
    sorted_pairs = sorted(zip(values, names, range(len(names))), reverse=True)
    sorted_values = [p[0] for p in sorted_pairs]
    sorted_names = [p[1] for p in sorted_pairs]
    sorted_indices = [p[2] for p in sorted_pairs]

    rects = _squarify(sorted_values, 0, 0, width, height)

    fig, ax = plt.subplots(1, 1, figsize=(width / 100, height / 100))
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')

    for i, (rx, ry, rw, rh) in enumerate(rects):
        color = palette[sorted_indices[i] % len(palette)]
        rect = mpatches.FancyBboxPatch(
            (rx + 1, ry + 1), rw - 2, rh - 2,
            boxstyle="round,pad=2",
            facecolor=color, edgecolor='white', linewidth=2, alpha=0.85
        )
        ax.add_patch(rect)
        # Label
        if rw > 30 and rh > 20:
            ax.text(
                rx + rw / 2, ry + rh / 2, f"{sorted_names[i]}\n{sorted_values[i]}",
                ha='center', va='center', fontsize=8, color='white', fontweight='bold'
            )

    ax.invert_yaxis()
    return _fig_to_svg(fig)


def voronoi_treemap(data: dict, title: str = '', width: int = 700, height: int = 700, colors: list = None) -> str:
    """Weighted Voronoi treemap using Lloyd's relaxation."""
    palette = _get_colors(colors)

    children = data.get('children', [])
    if not children:
        fig, ax = plt.subplots(1, 1, figsize=(width / 100, height / 100))
        ax.axis('off')
        return _fig_to_svg(fig)

    names = [c.get('name', '') for c in children]
    values = np.array([_get_node_value(c) for c in children], dtype=float)
    n = len(values)
    weights = values / values.sum()

    # Initialize points with weighted random placement
    rng = np.random.default_rng(42)
    points = rng.random((n, 2)) * 0.8 + 0.1  # Keep within [0.1, 0.9]

    # Lloyd's relaxation iterations
    for _ in range(50):
        # Add mirror points for bounded Voronoi
        mirrored = np.vstack([
            points,
            np.column_stack([-points[:, 0], points[:, 1]]),
            np.column_stack([2 - points[:, 0], points[:, 1]]),
            np.column_stack([points[:, 0], -points[:, 1]]),
            np.column_stack([points[:, 0], 2 - points[:, 1]]),
        ])
        try:
            vor = Voronoi(mirrored)
        except Exception:
            break

        new_points = np.copy(points)
        for i in range(n):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            if -1 in region or not region:
                continue
            polygon = vor.vertices[region]
            # Clip to [0, 1]
            polygon = np.clip(polygon, 0, 1)
            if len(polygon) >= 3:
                centroid = polygon.mean(axis=0)
                # Weighted move towards centroid
                new_points[i] = points[i] + 0.5 * (centroid - points[i])
                new_points[i] = np.clip(new_points[i], 0.05, 0.95)
        points = new_points

    # Final Voronoi for drawing
    mirrored = np.vstack([
        points,
        np.column_stack([-points[:, 0], points[:, 1]]),
        np.column_stack([2 - points[:, 0], points[:, 1]]),
        np.column_stack([points[:, 0], -points[:, 1]]),
        np.column_stack([points[:, 0], 2 - points[:, 1]]),
    ])
    vor = Voronoi(mirrored)

    fig, ax = plt.subplots(1, 1, figsize=(width / 100, height / 100))
    ax.set_aspect('equal')
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')

    for i in range(n):
        region_idx = vor.point_region[i]
        region = vor.regions[region_idx]
        if -1 in region or not region:
            continue
        polygon = vor.vertices[region]
        # Clip polygon to [0, 1] box
        polygon = np.clip(polygon, 0, 1)
        if len(polygon) >= 3:
            color = palette[i % len(palette)]
            poly = plt.Polygon(polygon, facecolor=color, edgecolor='white', linewidth=2, alpha=0.85)
            ax.add_patch(poly)
            centroid = polygon.mean(axis=0)
            ax.text(centroid[0], centroid[1], names[i], ha='center', va='center',
                    fontsize=8, color='white', fontweight='bold')

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    return _fig_to_svg(fig)


if __name__ == '__main__':
    import numpy as np
    from scipy.cluster.hierarchy import linkage
    from scomp_link.utils.report_html import ScompLinkHTMLReport
    import os
    os.makedirs('tmp', exist_ok=True)

    report = ScompLinkHTMLReport('RAWGraphs Hierarchies Demo')
    np.random.seed(42)

    # circlepacking
    data = {'name': 'World', 'children': [
        {'name': 'Europe', 'children': [{'name': 'France', 'value': 67}, {'name': 'Germany', 'value': 83}, {'name': 'Italy', 'value': 60}]},
        {'name': 'Asia', 'children': [{'name': 'China', 'value': 140}, {'name': 'Japan', 'value': 126}]},
        {'name': 'Americas', 'children': [{'name': 'USA', 'value': 330}, {'name': 'Brazil', 'value': 210}]}
    ]}
    svg = circlepacking(data, 'World Population')
    report.add_rawgraphs_to_report(svg, 'Circle Packing')

    # dendrogram
    X = np.random.rand(8, 4)
    Z = linkage(X, method='ward')
    svg = dendrogram(Z, ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'], 'Cluster Hierarchy')
    report.add_rawgraphs_to_report(svg, 'Dendrogram')

    # circular_dendrogram
    svg = circular_dendrogram(Z, ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'], 'Circular Dendrogram')
    report.add_rawgraphs_to_report(svg, 'Circular Dendrogram')

    # sunburst
    svg = sunburst(data, 'Population Sunburst')
    report.add_rawgraphs_to_report(svg, 'Sunburst')

    # treemap
    tree_data = {'name': 'Budget', 'children': [
        {'name': 'Engineering', 'value': 40},
        {'name': 'Marketing', 'value': 25},
        {'name': 'Sales', 'value': 20},
        {'name': 'Support', 'value': 10},
        {'name': 'HR', 'value': 5}
    ]}
    svg = treemap(tree_data, 'Department Budget')
    report.add_rawgraphs_to_report(svg, 'Treemap')

    # voronoi_treemap
    svg = voronoi_treemap(tree_data, 'Budget Voronoi')
    report.add_rawgraphs_to_report(svg, 'Voronoi Treemap')

    report.save_html('tmp/demo_hierarchies.html')
    print('Saved tmp/demo_hierarchies.html')
