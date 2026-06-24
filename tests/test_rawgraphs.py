# -*- coding: utf-8 -*-
"""Tests for scomp_link.utils.rawgraphs and scomp_link.utils.colors."""
import numpy as np
import pytest
from scipy.cluster.hierarchy import linkage


class TestColors:
    """Test centralized color palette module."""

    def test_primary_palette_length(self):
        from scomp_link.utils.colors import PRIMARY
        assert len(PRIMARY) == 10

    def test_all_palettes_same_length(self):
        from scomp_link.utils.colors import PRIMARY, LIGHT, MEDIUM_LIGHT, MEDIUM, MEDIUM_DARK, DARK, DARKEST
        for pal in (LIGHT, MEDIUM_LIGHT, MEDIUM, MEDIUM_DARK, DARK, DARKEST):
            assert len(pal) == len(PRIMARY)

    def test_theme_colors_are_strings(self):
        from scomp_link.utils.colors import MAIN, MAIN_LIGHT, MAIN_DARK
        for c in (MAIN, MAIN_LIGHT, MAIN_DARK):
            assert isinstance(c, str)
            assert c.startswith('#')

    def test_primary_json_format(self):
        from scomp_link.utils.colors import PRIMARY_JSON
        import json
        parsed = json.loads(PRIMARY_JSON)
        assert len(parsed) == 10


class TestComparisons:
    """Test comparisons.py chart functions."""

    def test_barchart(self):
        from scomp_link.utils.rawgraphs import barchart
        svg = barchart(['A', 'B', 'C'], [10, 20, 30], 'Test')
        assert svg.startswith('<svg')
        assert '</svg>' in svg

    def test_barchartmultiset(self):
        from scomp_link.utils.rawgraphs import barchartmultiset
        svg = barchartmultiset(['A', 'B'], {'G1': [1, 2], 'G2': [3, 4]}, 'Test')
        assert '<svg' in svg

    def test_barchartstacked(self):
        from scomp_link.utils.rawgraphs import barchartstacked
        svg = barchartstacked(['A', 'B'], {'G1': [1, 2], 'G2': [3, 4]}, 'Test')
        assert '<svg' in svg

    def test_piechart(self):
        from scomp_link.utils.rawgraphs import piechart
        svg = piechart(['X', 'Y', 'Z'], [40, 35, 25], 'Test')
        assert '<svg' in svg

    def test_radarchart(self):
        from scomp_link.utils.rawgraphs import radarchart
        svg = radarchart(['A', 'B', 'C', 'D'], {'S1': [3, 5, 2, 4]}, 'Test')
        assert '<svg' in svg

    def test_voronoidiagram(self):
        from scomp_link.utils.rawgraphs import voronoidiagram
        pts = [(0, 0), (1, 0), (0.5, 1), (0, 1), (1, 1)]
        svg = voronoidiagram(pts, title='Test')
        assert '<svg' in svg


class TestDistributions:
    """Test distributions.py chart functions."""

    def test_beeswarm(self):
        from scomp_link.utils.rawgraphs import beeswarm
        svg = beeswarm([1, 2, 3, 4, 5, 6], ['A', 'A', 'A', 'B', 'B', 'B'], 'Test')
        assert '<svg' in svg

    def test_beeswarm_no_groups(self):
        from scomp_link.utils.rawgraphs import beeswarm
        svg = beeswarm([1, 2, 3, 4, 5])
        assert '<svg' in svg

    def test_boxplot(self):
        from scomp_link.utils.rawgraphs import boxplot
        svg = boxplot([[1, 2, 3, 4], [5, 6, 7, 8]], ['A', 'B'], 'Test')
        assert '<svg' in svg

    def test_violinplot(self):
        from scomp_link.utils.rawgraphs import violinplot
        svg = violinplot([[1, 2, 3, 4, 5], [3, 4, 5, 6, 7]], ['A', 'B'], 'Test')
        assert '<svg' in svg


class TestTimeSeries:
    """Test time_series.py chart functions."""

    def test_bumpchart(self):
        from scomp_link.utils.rawgraphs import bumpchart
        svg = bumpchart({'A': [1, 2, 3], 'B': [3, 1, 2]}, ['Q1', 'Q2', 'Q3'], 'Test')
        assert '<svg' in svg

    def test_gantt_chart(self):
        from scomp_link.utils.rawgraphs import gantt_chart
        tasks = [{'name': 'T1', 'start': 0, 'end': 3, 'group': 'P1'},
                 {'name': 'T2', 'start': 2, 'end': 5, 'group': 'P2'}]
        svg = gantt_chart(tasks, 'Test')
        assert '<svg' in svg

    def test_horizongraph(self):
        from scomp_link.utils.rawgraphs import horizongraph
        svg = horizongraph({'A': [1, -2, 3, -1, 2, 0]}, list(range(6)), 'Test')
        assert '<svg' in svg

    def test_linechart(self):
        from scomp_link.utils.rawgraphs import linechart
        svg = linechart({'A': [1, 2, 3], 'B': [3, 2, 1]}, ['Jan', 'Feb', 'Mar'], 'Test')
        assert '<svg' in svg

    def test_slopechart(self):
        from scomp_link.utils.rawgraphs import slopechart
        svg = slopechart({'A': [10, 20], 'B': [15, 12]}, ['Before', 'After'], 'Test')
        assert '<svg' in svg

    def test_streamgraph(self):
        from scomp_link.utils.rawgraphs import streamgraph
        svg = streamgraph({'A': [1, 2, 3, 4], 'B': [4, 3, 2, 1]}, list(range(4)), 'Test')
        assert '<svg' in svg


class TestCorrelations:
    """Test correlations.py chart functions."""

    def test_bubblechart(self):
        from scomp_link.utils.rawgraphs import bubblechart
        svg = bubblechart([1, 2, 3], [4, 5, 6], [10, 20, 30], title='Test')
        assert '<svg' in svg

    def test_contour_plot(self):
        from scomp_link.utils.rawgraphs import contour_plot
        np.random.seed(0)
        svg = contour_plot(np.random.normal(0, 1, 50).tolist(),
                           np.random.normal(0, 1, 50).tolist(), 'Test')
        assert '<svg' in svg

    def test_convex_hull(self):
        from scomp_link.utils.rawgraphs import convex_hull
        np.random.seed(0)
        x = np.concatenate([np.random.normal(0, 1, 15), np.random.normal(3, 1, 15)]).tolist()
        y = np.concatenate([np.random.normal(0, 1, 15), np.random.normal(3, 1, 15)]).tolist()
        svg = convex_hull(x, y, ['A'] * 15 + ['B'] * 15, 'Test')
        assert '<svg' in svg

    def test_hexagonal_binning(self):
        from scomp_link.utils.rawgraphs import hexagonal_binning
        np.random.seed(0)
        svg = hexagonal_binning(np.random.normal(0, 1, 100).tolist(),
                                np.random.normal(0, 1, 100).tolist(), 'Test')
        assert '<svg' in svg

    def test_matrixplot(self):
        from scomp_link.utils.rawgraphs import matrixplot
        svg = matrixplot([[1, 2], [3, 4]], ['r1', 'r2'], ['c1', 'c2'], 'Test')
        assert '<svg' in svg

    def test_parallelcoordinates(self):
        from scomp_link.utils.rawgraphs import parallelcoordinates
        data = {'A': [1, 2, 3], 'B': [3, 2, 1], 'C': ['x', 'y', 'x']}
        svg = parallelcoordinates(data, 'C', 'Test')
        assert '<svg' in svg


class TestHierarchies:
    """Test hierarchies.py chart functions."""

    def test_circlepacking(self):
        from scomp_link.utils.rawgraphs import circlepacking
        data = {'name': 'root', 'children': [{'name': 'A', 'value': 10}, {'name': 'B', 'value': 20}]}
        svg = circlepacking(data, 'Test')
        assert '<svg' in svg

    def test_dendrogram(self):
        from scomp_link.utils.rawgraphs import dendrogram
        np.random.seed(0)
        Z = linkage(np.random.rand(5, 3), method='ward')
        svg = dendrogram(Z, ['a', 'b', 'c', 'd', 'e'], 'Test')
        assert '<svg' in svg

    def test_circular_dendrogram(self):
        from scomp_link.utils.rawgraphs import circular_dendrogram
        np.random.seed(0)
        Z = linkage(np.random.rand(5, 3), method='ward')
        svg = circular_dendrogram(Z, ['a', 'b', 'c', 'd', 'e'], 'Test')
        assert '<svg' in svg

    def test_sunburst(self):
        from scomp_link.utils.rawgraphs import sunburst
        data = {'name': 'root', 'children': [{'name': 'A', 'value': 10}, {'name': 'B', 'value': 20}]}
        svg = sunburst(data, 'Test')
        assert '<svg' in svg

    def test_treemap(self):
        from scomp_link.utils.rawgraphs import treemap
        data = {'name': 'root', 'children': [{'name': 'A', 'value': 30}, {'name': 'B', 'value': 20}]}
        svg = treemap(data, 'Test')
        assert '<svg' in svg

    def test_voronoi_treemap(self):
        from scomp_link.utils.rawgraphs import voronoi_treemap
        data = {'name': 'root', 'children': [{'name': 'A', 'value': 10}, {'name': 'B', 'value': 20}]}
        svg = voronoi_treemap(data, 'Test')
        assert '<svg' in svg


class TestNetworks:
    """Test networks.py chart functions."""

    def test_alluvial_diagram(self):
        from scomp_link.utils.rawgraphs import alluvial_diagram
        flows = [{'source': 'A', 'target': 'X', 'value': 10},
                 {'source': 'B', 'target': 'Y', 'value': 5}]
        svg = alluvial_diagram(flows, 'Test')
        assert '<svg' in svg

    def test_arc_diagram(self):
        from scomp_link.utils.rawgraphs import arc_diagram
        svg = arc_diagram(['A', 'B', 'C'], [{'source': 0, 'target': 2, 'value': 5}], 'Test')
        assert '<svg' in svg

    def test_chord_diagram(self):
        from scomp_link.utils.rawgraphs import chord_diagram
        matrix = [[0, 5, 3], [5, 0, 4], [3, 4, 0]]
        svg = chord_diagram(matrix, ['A', 'B', 'C'], 'Test')
        assert '<svg' in svg

    def test_sankey_diagram(self):
        from scomp_link.utils.rawgraphs import sankey_diagram
        nodes = [{'name': 'S', 'x': 0}, {'name': 'T', 'x': 1}]
        links = [{'source': 0, 'target': 1, 'value': 10}]
        svg = sankey_diagram(nodes, links, 'Test')
        assert '<svg' in svg


class TestReportIntegration:
    """Test add_rawgraphs_to_report integration."""

    def test_add_rawgraphs_to_report(self):
        from scomp_link.utils.report_html import ScompLinkHTMLReport
        from scomp_link.utils.rawgraphs import barchart
        report = ScompLinkHTMLReport('Test Report')
        svg = barchart(['A', 'B'], [1, 2], 'Chart')
        report.add_rawgraphs_to_report(svg, 'My Chart')
        assert '<h2>My Chart</h2>' in report.html_report
        assert '<svg' in report.html_report

    def test_report_uses_colors_defaults(self):
        from scomp_link.utils.report_html import ScompLinkHTMLReport
        from scomp_link.utils.colors import MAIN
        report = ScompLinkHTMLReport('Test')
        assert report.main_color == MAIN
