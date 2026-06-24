# -*- coding: utf-8 -*-
"""
██████╗  █████╗ ██╗    ██╗ ██████╗ ██████╗  █████╗ ██████╗ ██╗  ██╗███████╗
██╔══██╗██╔══██╗██║    ██║██╔════╝ ██╔══██╗██╔══██╗██╔══██╗██║  ██║██╔════╝
██████╔╝███████║██║ █╗ ██║██║  ███╗██████╔╝███████║██████╔╝███████║███████╗
██╔══██╗██╔══██║██║███╗██║██║   ██║██╔══██╗██╔══██║██╔═══╝ ██╔══██║╚════██║
██║  ██║██║  ██║╚███╔███╔╝╚██████╔╝██║  ██║██║  ██║██║     ██║  ██║███████║
╚═╝  ╚═╝╚═╝  ╚═╝ ╚══╝╚══╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝     ╚═╝  ╚═╝╚══════╝

RAWGraphs-style distribution chart functions for scomp-link.
Generates SVG charts server-side using matplotlib.
"""

import io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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


def beeswarm(data: list, groups: list = None, title: str = '',
             width: int = 800, height: int = 500, colors: list = None) -> str:
    """
    Beeswarm (jittered strip) plot showing individual data points.

    :param data: list of numeric values
    :param groups: list of group labels (same length as data), None for single group
    :param title: chart title
    :param width: SVG width in pixels
    :param height: SVG height in pixels
    :param colors: custom color palette
    :return: SVG string

    Example:
        svg = beeswarm([1,2,3,4,5,6], ['A','A','A','B','B','B'], 'My Beeswarm')
    """
    palette = colors or COLORS
    data = np.asarray(data, dtype=float)
    if groups is None:
        groups = [''] * len(data)
    groups = np.asarray(groups)
    unique_groups = list(dict.fromkeys(groups))

    fig, ax = plt.subplots(figsize=(width / 100, height / 100))
    for i, g in enumerate(unique_groups):
        mask = groups == g
        vals = data[mask]
        x = np.full_like(vals, i, dtype=float)
        x += np.random.uniform(-0.25, 0.25, size=len(vals))
        ax.scatter(x, vals, color=palette[i % len(palette)], alpha=0.7,
                   edgecolors='white', linewidth=0.5, s=30)

    ax.set_xticks(range(len(unique_groups)))
    ax.set_xticklabels(unique_groups)
    ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return _fig_to_svg(fig)


def boxplot(data: list, labels: list = None, title: str = '',
            width: int = 800, height: int = 500, colors: list = None) -> str:
    """
    Box-and-whisker plot.

    :param data: list of lists (each inner list is a distribution for one group)
    :param labels: group labels
    :param title: chart title
    :param width: SVG width in pixels
    :param height: SVG height in pixels
    :param colors: custom color palette
    :return: SVG string

    Example:
        svg = boxplot([[1,2,3,4], [2,3,4,5]], ['Group A', 'Group B'], 'Boxplot')
    """
    palette = colors or COLORS
    fig, ax = plt.subplots(figsize=(width / 100, height / 100))
    bp = ax.boxplot(data, patch_artist=True, notch=False)
    if labels:
        ax.set_xticklabels(labels)
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(palette[i % len(palette)])
        patch.set_alpha(0.7)
    for median in bp['medians']:
        median.set_color('#333333')
    ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return _fig_to_svg(fig)


def violinplot(data: list, labels: list = None, title: str = '',
               width: int = 800, height: int = 500, colors: list = None) -> str:
    """
    Violin plot showing density distribution.

    :param data: list of lists (each inner list is a distribution for one group)
    :param labels: group labels
    :param title: chart title
    :param width: SVG width in pixels
    :param height: SVG height in pixels
    :param colors: custom color palette
    :return: SVG string

    Example:
        svg = violinplot([[1,2,3,4,5], [3,4,5,6,7]], ['A', 'B'], 'Violin')
    """
    palette = colors or COLORS
    fig, ax = plt.subplots(figsize=(width / 100, height / 100))
    vp = ax.violinplot(data, showmeans=False, showmedians=True)
    for i, body in enumerate(vp['bodies']):
        body.set_facecolor(palette[i % len(palette)])
        body.set_alpha(0.7)
    for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians'):
        if partname in vp:
            vp[partname].set_edgecolor('#333333')
    if labels:
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels)
    ax.set_title(title)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return _fig_to_svg(fig)


if __name__ == '__main__':
    from scomp_link.utils.report_html import ScompLinkHTMLReport
    import os
    os.makedirs('tmp', exist_ok=True)

    report = ScompLinkHTMLReport('RAWGraphs Distributions Demo')
    np.random.seed(42)

    # beeswarm
    data = np.concatenate([np.random.normal(5, 1, 50),
                           np.random.normal(8, 1.5, 50),
                           np.random.normal(6, 0.8, 50)])
    groups = ['A'] * 50 + ['B'] * 50 + ['C'] * 50
    svg = beeswarm(data.tolist(), groups, 'Score Distribution')
    report.add_rawgraphs_to_report(svg, 'Beeswarm Plot')

    # boxplot
    data_box = [np.random.normal(10, 2, 100).tolist(),
                np.random.normal(15, 3, 100).tolist(),
                np.random.normal(12, 1.5, 100).tolist()]
    svg = boxplot(data_box, ['Control', 'Treatment A', 'Treatment B'], 'Experiment Results')
    report.add_rawgraphs_to_report(svg, 'Box Plot')

    # violinplot
    svg = violinplot(data_box, ['Control', 'Treatment A', 'Treatment B'], 'Density Comparison')
    report.add_rawgraphs_to_report(svg, 'Violin Plot')

    report.save_html('tmp/demo_distributions.html')
    print('Saved tmp/demo_distributions.html')
