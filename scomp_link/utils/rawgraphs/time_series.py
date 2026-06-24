# -*- coding: utf-8 -*-
"""
██████╗  █████╗ ██╗    ██╗ ██████╗ ██████╗  █████╗ ██████╗ ██╗  ██╗███████╗
██╔══██╗██╔══██╗██║    ██║██╔════╝ ██╔══██╗██╔══██╗██╔══██╗██║  ██║██╔════╝
██████╔╝███████║██║ █╗ ██║██║  ███╗██████╔╝███████║██████╔╝███████║███████╗
██╔══██╗██╔══██║██║███╗██║██║   ██║██╔══██╗██╔══██║██╔═══╝ ██╔══██║╚════██║
██║  ██║██║  ██║╚███╔███╔╝╚██████╔╝██║  ██║██║  ██║██║     ██║  ██║███████║
╚═╝  ╚═╝╚═╝  ╚═╝ ╚══╝╚══╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝╚══════╝

RAWGraphs-style time series chart functions for scomp-link.
Generates SVG charts server-side using matplotlib.
"""

import io
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

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


def bumpchart(ranks: dict, periods: list, title: str = '', width: int = 800, height: int = 500, colors: list = None) -> str:
    """Bump chart showing ranking changes over time. Y-axis inverted (rank 1 at top)."""
    colors = colors or COLORS
    fig, ax = plt.subplots(figsize=(width / 100, height / 100))

    x = list(range(len(periods)))
    for i, (name, values) in enumerate(ranks.items()):
        c = colors[i % len(colors)]
        ax.plot(x, values, marker='o', markersize=10, linewidth=2.5, color=c, label=name)
        ax.annotate(name, xy=(x[-1], values[-1]), xytext=(5, 0),
                    textcoords='offset points', va='center', fontsize=9, color=c, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels(periods)
    ax.invert_yaxis()
    ax.set_yticks(sorted(set(v for vals in ranks.values() for v in vals)))
    ax.set_ylabel('Rank')
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    return _fig_to_svg(fig)


def gantt_chart(tasks: list, title: str = '', width: int = 900, height: int = 500, colors: list = None) -> str:
    """Gantt chart for project timelines using broken_barh."""
    colors = colors or COLORS
    fig, ax = plt.subplots(figsize=(width / 100, height / 100))

    # Assign colors by group
    groups = list(dict.fromkeys(t.get('group', '') for t in tasks))
    group_colors = {g: colors[i % len(colors)] for i, g in enumerate(groups)}

    y_ticks = []
    y_labels = []
    for i, task in enumerate(tasks):
        y_pos = len(tasks) - 1 - i
        duration = task['end'] - task['start']
        c = group_colors.get(task.get('group', ''), colors[0])
        ax.broken_barh([(task['start'], duration)], (y_pos - 0.35, 0.7), facecolors=c, edgecolors='white', linewidth=0.5)
        y_ticks.append(y_pos)
        y_labels.append(task['name'])

    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_xlabel('Time')
    ax.grid(axis='x', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Legend for groups
    handles = [mpatches.Patch(color=group_colors[g], label=g) for g in groups if g]
    if handles:
        ax.legend(handles=handles, loc='lower right', fontsize=8)

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    return _fig_to_svg(fig)


def horizongraph(series: dict, x_values: list, title: str = '', width: int = 800, height: int = 400, colors: list = None, bands: int = 3) -> str:
    """Horizon graph: layered bands showing positive/negative values with increasing opacity."""
    colors = colors or COLORS
    n_series = len(series)
    fig, axes = plt.subplots(n_series, 1, figsize=(width / 100, height / 100), sharex=True)
    if n_series == 1:
        axes = [axes]

    x = np.array(x_values)
    for idx, (name, values) in enumerate(series.items()):
        ax = axes[idx]
        vals = np.array(values, dtype=float)
        max_abs = max(abs(vals.min()), abs(vals.max())) or 1
        band_size = max_abs / bands
        base_color = colors[idx % len(colors)]

        # Convert hex to RGB
        r, g, b = int(base_color[1:3], 16) / 255, int(base_color[3:5], 16) / 255, int(base_color[5:7], 16) / 255

        for band_i in range(bands):
            lower = band_i * band_size
            upper = (band_i + 1) * band_size
            alpha = (band_i + 1) / bands

            # Positive bands
            pos_clipped = np.clip(vals, lower, upper) - lower
            pos_clipped = np.where(vals > lower, pos_clipped, 0)
            ax.fill_between(x, 0, pos_clipped, color=(r, g, b, alpha))

            # Negative bands
            neg_vals = -vals
            neg_clipped = np.clip(neg_vals, lower, upper) - lower
            neg_clipped = np.where(neg_vals > lower, neg_clipped, 0)
            ax.fill_between(x, 0, neg_clipped, color=(1 - r, 1 - g, 1 - b, alpha))

        ax.set_ylim(0, band_size)
        ax.set_ylabel(name, fontsize=8, rotation=0, labelpad=50, va='center')
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)

    if title:
        axes[0].set_title(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    return _fig_to_svg(fig)


def linechart(series: dict, x_values: list, title: str = '', width: int = 800, height: int = 500, colors: list = None) -> str:
    """Multi-line chart."""
    colors = colors or COLORS
    fig, ax = plt.subplots(figsize=(width / 100, height / 100))

    x = list(range(len(x_values)))
    for i, (name, values) in enumerate(series.items()):
        ax.plot(x, values, marker='o', markersize=5, linewidth=2, color=colors[i % len(colors)], label=name)

    ax.set_xticks(x)
    ax.set_xticklabels(x_values)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    return _fig_to_svg(fig)


def slopechart(data: dict, period_labels: list = None, title: str = '', width: int = 600, height: int = 500, colors: list = None) -> str:
    """Slope chart comparing values between two points (before/after)."""
    colors = colors or COLORS
    period_labels = period_labels or ['Before', 'After']
    fig, ax = plt.subplots(figsize=(width / 100, height / 100))

    for i, (name, values) in enumerate(data.items()):
        c = colors[i % len(colors)]
        ax.plot([0, 1], values, marker='o', markersize=8, linewidth=2.5, color=c)
        ax.annotate(f'{name} ({values[0]})', xy=(0, values[0]), xytext=(-10, 0),
                    textcoords='offset points', ha='right', va='center', fontsize=9, color=c, fontweight='bold')
        ax.annotate(f'{name} ({values[1]})', xy=(1, values[1]), xytext=(10, 0),
                    textcoords='offset points', ha='left', va='center', fontsize=9, color=c, fontweight='bold')

    ax.set_xticks([0, 1])
    ax.set_xticklabels(period_labels, fontsize=12, fontweight='bold')
    ax.set_xlim(-0.3, 1.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='x', length=0)
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    return _fig_to_svg(fig)


def streamgraph(series: dict, x_values: list, title: str = '', width: int = 800, height: int = 500, colors: list = None) -> str:
    """Stacked area chart centered on baseline (streamgraph)."""
    colors = colors or COLORS
    fig, ax = plt.subplots(figsize=(width / 100, height / 100))

    x = np.array(x_values)
    labels = list(series.keys())
    y = np.array([series[k] for k in labels])
    c = [colors[i % len(colors)] for i in range(len(labels))]

    ax.stackplot(x, y, labels=labels, colors=c, baseline='sym')
    ax.legend(loc='upper left', fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.3)
    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    fig.tight_layout()
    return _fig_to_svg(fig)


if __name__ == '__main__':
    import numpy as np
    from scomp_link.utils.report_html import ScompLinkHTMLReport
    import os
    os.makedirs('tmp', exist_ok=True)

    report = ScompLinkHTMLReport('RAWGraphs Time Series Demo')
    np.random.seed(42)

    # bumpchart
    svg = bumpchart({'Apple': [1, 2, 1, 1], 'Google': [2, 1, 3, 2], 'Microsoft': [3, 3, 2, 3]},
                    ['2020', '2021', '2022', '2023'], 'Market Rank')
    report.add_rawgraphs_to_report(svg, 'Bump Chart')

    # gantt_chart
    tasks = [{'name': 'Design', 'start': 0, 'end': 3, 'group': 'Phase 1'},
             {'name': 'Develop', 'start': 2, 'end': 7, 'group': 'Phase 1'},
             {'name': 'Test', 'start': 6, 'end': 9, 'group': 'Phase 2'},
             {'name': 'Deploy', 'start': 8, 'end': 10, 'group': 'Phase 2'}]
    svg = gantt_chart(tasks, 'Project Timeline')
    report.add_rawgraphs_to_report(svg, 'Gantt Chart')

    # horizongraph
    x = list(range(50))
    svg = horizongraph({'Temperature': (np.sin(np.linspace(0, 4 * np.pi, 50)) * 10).tolist(),
                        'Pressure': (np.cos(np.linspace(0, 3 * np.pi, 50)) * 5).tolist()}, x, 'Sensor Data')
    report.add_rawgraphs_to_report(svg, 'Horizon Graph')

    # linechart
    svg = linechart({'Revenue': [10, 15, 13, 18, 22, 25], 'Costs': [8, 9, 11, 12, 14, 15]},
                    ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'], 'Financial Trend')
    report.add_rawgraphs_to_report(svg, 'Line Chart')

    # slopechart
    svg = slopechart({'Product A': [45, 62], 'Product B': [70, 55], 'Product C': [30, 48]},
                     ['2022', '2023'], 'Year-over-Year Change')
    report.add_rawgraphs_to_report(svg, 'Slope Chart')

    # streamgraph
    x = list(range(20))
    svg = streamgraph({'Rock': np.random.randint(5, 20, 20).tolist(),
                       'Pop': np.random.randint(3, 15, 20).tolist(),
                       'Jazz': np.random.randint(2, 10, 20).tolist()}, x, 'Music Trends')
    report.add_rawgraphs_to_report(svg, 'Streamgraph')

    report.save_html('tmp/demo_time_series.html')
    print('Saved tmp/demo_time_series.html')
