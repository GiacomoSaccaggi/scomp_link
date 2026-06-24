# -*- coding: utf-8 -*-
"""
██████╗  █████╗ ██╗    ██╗ ██████╗ ██████╗  █████╗ ██████╗ ██╗  ██╗███████╗
██╔══██╗██╔══██╗██║    ██║██╔════╝ ██╔══██╗██╔══██╗██╔══██╗██║  ██║██╔════╝
██████╔╝███████║██║ █╗ ██║██║  ███╗██████╔╝███████║██████╔╝███████║███████╗
██╔══██╗██╔══██║██║███╗██║██║   ██║██╔══██╗██╔══██║██╔═══╝ ██╔══██║╚════██║
██║  ██║██║  ██║╚███╔███╔╝╚██████╔╝██║  ██║██║  ██║██║     ██║  ██║███████║
╚═╝  ╚═╝╚═╝  ╚═╝ ╚══╝╚══╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝╚══════╝

RAWGraphs-style network/flow chart functions for scomp-link.
Generates SVG charts programmatically using numpy for geometry.
"""

import numpy as np
from collections import defaultdict

from scomp_link.utils.colors import PRIMARY as COLORS


def _svg_start(width, height):
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}" style="max-width:100%;height:auto;">'


def _svg_end():
    return '</svg>'


def _color(i, colors):
    palette = colors or COLORS
    return palette[i % len(palette)]


def alluvial_diagram(flows: list, title: str = '', width: int = 900, height: int = 500, colors: list = None) -> str:
    """Alluvial/flow diagram showing connections between categories across stages."""
    margin = {'top': 50, 'bottom': 20, 'left': 120, 'right': 120}
    bar_width = 20
    node_gap = 10
    draw_h = height - margin['top'] - margin['bottom']

    # Collect unique sources and targets
    sources = list(dict.fromkeys(f['source'] for f in flows))
    targets = list(dict.fromkeys(f['target'] for f in flows))

    # Calculate totals per node
    source_totals = defaultdict(float)
    target_totals = defaultdict(float)
    for f in flows:
        source_totals[f['source']] += f['value']
        target_totals[f['target']] += f['value']

    # Compute vertical positions proportional to value
    total_source = sum(source_totals[s] for s in sources)
    total_target = sum(target_totals[t] for t in targets)
    avail_h = draw_h - node_gap * (max(len(sources), len(targets)) - 1)

    # Source node positions
    src_pos = {}
    y = margin['top']
    for s in sources:
        h = (source_totals[s] / total_source) * avail_h
        src_pos[s] = (y, h)
        y += h + node_gap

    # Target node positions
    tgt_pos = {}
    y = margin['top']
    for t in targets:
        h = (target_totals[t] / total_target) * avail_h
        tgt_pos[t] = (y, h)
        y += h + node_gap

    x_src = margin['left']
    x_tgt = width - margin['right'] - bar_width

    parts = [_svg_start(width, height)]

    # Title
    if title:
        parts.append(f'<text x="{width/2}" y="30" font-size="16" font-weight="bold" text-anchor="middle" fill="#333">{title}</text>')

    # Track offsets for stacking flows within each node
    src_offset = {s: 0.0 for s in sources}
    tgt_offset = {t: 0.0 for t in targets}

    # Draw flow bands
    for i, f in enumerate(flows):
        s, t, val = f['source'], f['target'], f['value']
        s_y, s_h = src_pos[s]
        t_y, t_h = tgt_pos[t]

        band_h_src = (val / source_totals[s]) * s_h
        band_h_tgt = (val / target_totals[t]) * t_h

        y0_top = s_y + src_offset[s]
        y0_bot = y0_top + band_h_src
        y1_top = t_y + tgt_offset[t]
        y1_bot = y1_top + band_h_tgt

        src_offset[s] += band_h_src
        tgt_offset[t] += band_h_tgt

        x0 = x_src + bar_width
        x1 = x_tgt
        cx = (x0 + x1) / 2

        # Closed band path: top curve forward, bottom curve back
        d = (f'M {x0},{y0_top} C {cx},{y0_top} {cx},{y1_top} {x1},{y1_top} '
             f'L {x1},{y1_bot} C {cx},{y1_bot} {cx},{y0_bot} {x0},{y0_bot} Z')
        color = _color(i, colors)
        parts.append(f'<path d="{d}" fill="{color}" fill-opacity="0.5" stroke="{color}" stroke-width="0.5"/>')

    # Draw source bars
    for i, s in enumerate(sources):
        y, h = src_pos[s]
        color = _color(i, colors)
        parts.append(f'<rect x="{x_src}" y="{y}" width="{bar_width}" height="{h}" fill="{color}" rx="3"/>')
        parts.append(f'<text x="{x_src - 5}" y="{y + h/2 + 4}" font-size="11" text-anchor="end" fill="#333">{s}</text>')

    # Draw target bars
    for i, t in enumerate(targets):
        y, h = tgt_pos[t]
        color = _color(len(sources) + i, colors)
        parts.append(f'<rect x="{x_tgt}" y="{y}" width="{bar_width}" height="{h}" fill="{color}" rx="3"/>')
        parts.append(f'<text x="{x_tgt + bar_width + 5}" y="{y + h/2 + 4}" font-size="11" text-anchor="start" fill="#333">{t}</text>')

    parts.append(_svg_end())
    return '\n'.join(parts)


def arc_diagram(nodes: list, links: list, title: str = '', width: int = 900, height: int = 400, colors: list = None) -> str:
    """Arc diagram: nodes on a horizontal line, connections as arcs above."""
    margin = {'left': 60, 'right': 60, 'top': 50, 'bottom': 60}
    n = len(nodes)
    if n == 0:
        return _svg_start(width, height) + _svg_end()

    baseline_y = height - margin['bottom']
    avail_w = width - margin['left'] - margin['right']
    spacing = avail_w / max(n - 1, 1)

    # Node x positions
    xs = [margin['left'] + i * spacing for i in range(n)]

    # Max value for scaling stroke width
    max_val = max((l['value'] for l in links), default=1)

    parts = [_svg_start(width, height)]

    if title:
        parts.append(f'<text x="{width/2}" y="30" font-size="16" font-weight="bold" text-anchor="middle" fill="#333">{title}</text>')

    # Draw arcs
    for i, link in enumerate(links):
        src, tgt, val = link['source'], link['target'], link['value']
        x1, x2 = xs[src], xs[tgt]
        if x1 > x2:
            x1, x2 = x2, x1
        mid_x = (x1 + x2) / 2
        radius = (x2 - x1) / 2
        sw = max(1, (val / max_val) * 6)
        color = _color(i, colors)
        # Semi-circular arc using SVG arc command
        d = f'M {x1},{baseline_y} A {radius},{radius} 0 0,1 {x2},{baseline_y}'
        parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{sw:.1f}" stroke-opacity="0.7"/>')

    # Draw nodes
    for i, name in enumerate(nodes):
        color = _color(i, colors)
        parts.append(f'<circle cx="{xs[i]}" cy="{baseline_y}" r="8" fill="{color}"/>')
        parts.append(f'<text x="{xs[i]}" y="{baseline_y + 25}" font-size="10" text-anchor="middle" fill="#333">{name}</text>')

    parts.append(_svg_end())
    return '\n'.join(parts)


def chord_diagram(matrix: list, labels: list = None, title: str = '', width: int = 700, height: int = 700, colors: list = None) -> str:
    """Chord diagram showing flows between groups arranged in a circle."""
    mat = np.array(matrix, dtype=float)
    n = mat.shape[0]
    if labels is None:
        labels = [f'G{i}' for i in range(n)]

    cx, cy = width / 2, height / 2
    outer_r = min(width, height) / 2 - 60
    inner_r = outer_r - 20
    gap = 0.02  # radians between groups

    # Total flow per group (sum of row + column)
    group_totals = mat.sum(axis=1) + mat.sum(axis=0)
    grand_total = group_totals.sum()
    if grand_total == 0:
        return _svg_start(width, height) + _svg_end()

    # Angle allocation
    avail_angle = 2 * np.pi - n * gap
    angles = (group_totals / grand_total) * avail_angle

    # Start angles for each group
    starts = np.zeros(n)
    cur = 0.0
    for i in range(n):
        starts[i] = cur
        cur += angles[i] + gap

    parts = [_svg_start(width, height)]

    if title:
        parts.append(f'<text x="{width/2}" y="30" font-size="16" font-weight="bold" text-anchor="middle" fill="#333">{title}</text>')

    # Draw outer arcs (group segments)
    for i in range(n):
        a0, a1 = starts[i], starts[i] + angles[i]
        color = _color(i, colors)
        # Outer arc path
        x0_o, y0_o = cx + outer_r * np.cos(a0), cy + outer_r * np.sin(a0)
        x1_o, y1_o = cx + outer_r * np.cos(a1), cy + outer_r * np.sin(a1)
        x0_i, y0_i = cx + inner_r * np.cos(a1), cy + inner_r * np.sin(a1)
        x1_i, y1_i = cx + inner_r * np.cos(a0), cy + inner_r * np.sin(a0)
        large = 1 if (a1 - a0) > np.pi else 0
        d = (f'M {x0_o:.1f},{y0_o:.1f} A {outer_r:.1f},{outer_r:.1f} 0 {large},1 {x1_o:.1f},{y1_o:.1f} '
             f'L {x0_i:.1f},{y0_i:.1f} A {inner_r:.1f},{inner_r:.1f} 0 {large},0 {x1_i:.1f},{y1_i:.1f} Z')
        parts.append(f'<path d="{d}" fill="{color}"/>')

        # Label
        mid_a = (a0 + a1) / 2
        lx = cx + (outer_r + 15) * np.cos(mid_a)
        ly = cy + (outer_r + 15) * np.sin(mid_a)
        anchor = 'start' if np.cos(mid_a) >= 0 else 'end'
        parts.append(f'<text x="{lx:.1f}" y="{ly:.1f}" font-size="11" text-anchor="{anchor}" fill="#333">{labels[i]}</text>')

    # Draw chords - track sub-offsets within each group arc
    offsets = np.zeros(n)
    for i in range(n):
        for j in range(i + 1, n):
            val_ij = mat[i][j] + mat[j][i]
            if val_ij == 0:
                continue
            # Angle span for this chord at group i
            span_i = (val_ij / group_totals[i]) * angles[i] if group_totals[i] > 0 else 0
            span_j = (val_ij / group_totals[j]) * angles[j] if group_totals[j] > 0 else 0

            a_i0 = starts[i] + offsets[i]
            a_i1 = a_i0 + span_i
            a_j0 = starts[j] + offsets[j]
            a_j1 = a_j0 + span_j

            offsets[i] += span_i
            offsets[j] += span_j

            # Chord as two quadratic bezier curves through center
            xi0 = cx + inner_r * np.cos(a_i0)
            yi0 = cy + inner_r * np.sin(a_i0)
            xi1 = cx + inner_r * np.cos(a_i1)
            yi1 = cy + inner_r * np.sin(a_i1)
            xj0 = cx + inner_r * np.cos(a_j0)
            yj0 = cy + inner_r * np.sin(a_j0)
            xj1 = cx + inner_r * np.cos(a_j1)
            yj1 = cy + inner_r * np.sin(a_j1)

            d = (f'M {xi0:.1f},{yi0:.1f} Q {cx:.1f},{cy:.1f} {xj0:.1f},{yj0:.1f} '
                 f'A {inner_r:.1f},{inner_r:.1f} 0 0,1 {xj1:.1f},{yj1:.1f} '
                 f'Q {cx:.1f},{cy:.1f} {xi1:.1f},{yi1:.1f} '
                 f'A {inner_r:.1f},{inner_r:.1f} 0 0,1 {xi0:.1f},{yi0:.1f} Z')
            color = _color(i, colors)
            parts.append(f'<path d="{d}" fill="{color}" fill-opacity="0.4" stroke="{color}" stroke-width="0.5"/>')

    parts.append(_svg_end())
    return '\n'.join(parts)


def sankey_diagram(nodes: list, links: list, title: str = '', width: int = 900, height: int = 500, colors: list = None) -> str:
    """Sankey flow diagram with nodes arranged in columns by x-position."""
    margin = {'top': 50, 'bottom': 20, 'left': 40, 'right': 40}
    node_width = 20
    node_gap = 15
    draw_h = height - margin['top'] - margin['bottom']
    draw_w = width - margin['left'] - margin['right']

    # Assign x if missing (topological order)
    nodes_list = [dict(n) for n in nodes]
    if not any('x' in n for n in nodes_list):
        # Simple topological assignment: BFS from nodes with no incoming links
        incoming = defaultdict(set)
        outgoing = defaultdict(set)
        for l in links:
            incoming[l['target']].add(l['source'])
            outgoing[l['source']].add(l['target'])
        assigned = {}
        queue = [i for i in range(len(nodes_list)) if i not in incoming or len(incoming[i]) == 0]
        for i in queue:
            assigned[i] = 0
        while queue:
            nxt = []
            for src in queue:
                for l in links:
                    if l['source'] == src:
                        t = l['target']
                        new_x = assigned[src] + 1
                        if t not in assigned or assigned[t] < new_x:
                            assigned[t] = new_x
                            nxt.append(t)
            queue = nxt
        for i, n in enumerate(nodes_list):
            n['x'] = assigned.get(i, 0)

    # Group by column
    columns = defaultdict(list)
    for i, n in enumerate(nodes_list):
        columns[n['x']].append(i)

    num_cols = max(columns.keys()) + 1
    col_spacing = (draw_w - node_width) / max(num_cols - 1, 1)

    # Calculate total flow per node
    node_totals = [0.0] * len(nodes_list)
    for l in links:
        node_totals[l['source']] = max(node_totals[l['source']], node_totals[l['source']])
        node_totals[l['target']] = max(node_totals[l['target']], node_totals[l['target']])
    # Recompute: total = max(sum_in, sum_out)
    sum_out = [0.0] * len(nodes_list)
    sum_in = [0.0] * len(nodes_list)
    for l in links:
        sum_out[l['source']] += l['value']
        sum_in[l['target']] += l['value']
    for i in range(len(nodes_list)):
        node_totals[i] = max(sum_out[i], sum_in[i])

    # Position nodes within each column
    node_y = [0.0] * len(nodes_list)
    node_h = [0.0] * len(nodes_list)
    for col_x, col_nodes in columns.items():
        col_total = sum(node_totals[i] for i in col_nodes)
        avail = draw_h - node_gap * (len(col_nodes) - 1)
        y = margin['top']
        for idx in col_nodes:
            h = (node_totals[idx] / col_total) * avail if col_total > 0 else avail / len(col_nodes)
            node_y[idx] = y
            node_h[idx] = h
            y += h + node_gap

    # X positions
    node_x = [0.0] * len(nodes_list)
    for i, n in enumerate(nodes_list):
        node_x[i] = margin['left'] + n['x'] * col_spacing

    parts = [_svg_start(width, height)]

    if title:
        parts.append(f'<text x="{width/2}" y="30" font-size="16" font-weight="bold" text-anchor="middle" fill="#333">{title}</text>')

    # Track flow offsets for stacking
    out_offset = [0.0] * len(nodes_list)
    in_offset = [0.0] * len(nodes_list)

    # Draw flow paths
    for i, l in enumerate(links):
        src, tgt, val = l['source'], l['target'], l['value']
        # Band height at source and target
        band_src = (val / node_totals[src]) * node_h[src] if node_totals[src] > 0 else 0
        band_tgt = (val / node_totals[tgt]) * node_h[tgt] if node_totals[tgt] > 0 else 0

        y0_top = node_y[src] + out_offset[src]
        y0_bot = y0_top + band_src
        y1_top = node_y[tgt] + in_offset[tgt]
        y1_bot = y1_top + band_tgt

        out_offset[src] += band_src
        in_offset[tgt] += band_tgt

        x0 = node_x[src] + node_width
        x1 = node_x[tgt]
        cx_mid = (x0 + x1) / 2

        d = (f'M {x0:.1f},{y0_top:.1f} C {cx_mid:.1f},{y0_top:.1f} {cx_mid:.1f},{y1_top:.1f} {x1:.1f},{y1_top:.1f} '
             f'L {x1:.1f},{y1_bot:.1f} C {cx_mid:.1f},{y1_bot:.1f} {cx_mid:.1f},{y0_bot:.1f} {x0:.1f},{y0_bot:.1f} Z')
        color = _color(src, colors)
        parts.append(f'<path d="{d}" fill="{color}" fill-opacity="0.45" stroke="{color}" stroke-width="0.5"/>')

    # Draw node rectangles
    for i, n in enumerate(nodes_list):
        color = _color(i, colors)
        parts.append(f'<rect x="{node_x[i]:.1f}" y="{node_y[i]:.1f}" width="{node_width}" height="{node_h[i]:.1f}" fill="{color}" rx="3"/>')
        # Label to the right of rightmost column, left of others
        lx = node_x[i] + node_width + 5
        parts.append(f'<text x="{lx:.1f}" y="{node_y[i] + node_h[i]/2 + 4:.1f}" font-size="10" text-anchor="start" fill="#333">{n["name"]}</text>')

    parts.append(_svg_end())
    return '\n'.join(parts)


if __name__ == '__main__':
    import numpy as np
    from scomp_link.utils.report_html import ScompLinkHTMLReport
    import os
    os.makedirs('tmp', exist_ok=True)

    report = ScompLinkHTMLReport('RAWGraphs Networks Demo')

    # alluvial_diagram
    flows = [
        {'source': 'Homepage', 'target': 'Products', 'value': 40},
        {'source': 'Homepage', 'target': 'About', 'value': 20},
        {'source': 'Homepage', 'target': 'Blog', 'value': 15},
        {'source': 'Products', 'target': 'Cart', 'value': 30},
        {'source': 'Products', 'target': 'Exit', 'value': 10},
        {'source': 'Blog', 'target': 'Products', 'value': 8},
    ]
    svg = alluvial_diagram(flows, 'User Flow')
    report.add_rawgraphs_to_report(svg, 'Alluvial Diagram')

    # arc_diagram
    nodes = ['Alice', 'Bob', 'Carol', 'Dave', 'Eve']
    links = [{'source': 0, 'target': 1, 'value': 5}, {'source': 0, 'target': 3, 'value': 3},
             {'source': 1, 'target': 2, 'value': 4}, {'source': 2, 'target': 4, 'value': 2},
             {'source': 3, 'target': 4, 'value': 6}]
    svg = arc_diagram(nodes, links, 'Communication Network')
    report.add_rawgraphs_to_report(svg, 'Arc Diagram')

    # chord_diagram
    matrix = [[0, 5, 3, 2], [5, 0, 4, 1], [3, 4, 0, 6], [2, 1, 6, 0]]
    svg = chord_diagram(matrix, ['Dept A', 'Dept B', 'Dept C', 'Dept D'], 'Inter-department Flow')
    report.add_rawgraphs_to_report(svg, 'Chord Diagram')

    # sankey_diagram
    nodes = [{'name': 'Solar', 'x': 0}, {'name': 'Wind', 'x': 0}, {'name': 'Grid', 'x': 1},
             {'name': 'Industry', 'x': 2}, {'name': 'Residential', 'x': 2}]
    links = [{'source': 0, 'target': 2, 'value': 40}, {'source': 1, 'target': 2, 'value': 30},
             {'source': 2, 'target': 3, 'value': 45}, {'source': 2, 'target': 4, 'value': 25}]
    svg = sankey_diagram(nodes, links, 'Energy Flow')
    report.add_rawgraphs_to_report(svg, 'Sankey Diagram')

    report.save_html('tmp/demo_networks.html')
    print('Saved tmp/demo_networks.html')
