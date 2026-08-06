"""Tests for add_cascading_content, add_dataframe thresholds, plotly utils, and deprecation wrappers."""

import warnings

import numpy as np
import pandas as pd
import pytest


class TestCascadingContent:
    """Tests for ScompLinkHTMLReport.add_cascading_content."""

    def _make_report(self):
        from scomp_link.utils.report_html import ScompLinkHTMLReport

        return ScompLinkHTMLReport("Test Report")

    def test_single_dimension(self):
        report = self._make_report()
        dims = [{"label": "Region", "options": ["North", "South"]}]
        content = {
            ("North",): "<p>North content</p>",
            ("South",): "<p>South content</p>",
        }
        report.add_cascading_content("Test", dims, content)
        html = report.html_report
        assert "North content" in html
        assert "South content" in html
        assert "display:block" in html or "display: block" in html
        assert html.count("display:none") >= 1 or html.count("display: none") >= 1
        assert "<select" in html
        assert "Region" in html

    def test_two_dimensions(self):
        report = self._make_report()
        dims = [
            {"label": "Region", "options": ["North", "South"]},
            {"label": "Year", "options": ["2024", "2025"]},
        ]
        content = {
            ("North", "2024"): "<p>N-24</p>",
            ("North", "2025"): "<p>N-25</p>",
            ("South", "2024"): "<p>S-24</p>",
            ("South", "2025"): "<p>S-25</p>",
        }
        report.add_cascading_content("Test", dims, content)
        html = report.html_report
        assert "N-24" in html
        assert "S-25" in html
        assert html.count("<select") == 2

    def test_three_dimensions(self):
        report = self._make_report()
        dims = [
            {"label": "A", "options": ["a1", "a2"]},
            {"label": "B", "options": ["b1"]},
            {"label": "C", "options": ["c1", "c2"]},
        ]
        content = {
            ("a1", "b1", "c1"): "content_a1b1c1",
            ("a1", "b1", "c2"): "content_a1b1c2",
            ("a2", "b1", "c1"): "content_a2b1c1",
            ("a2", "b1", "c2"): "content_a2b1c2",
        }
        report.add_cascading_content("Test", dims, content)
        html = report.html_report
        assert html.count("<select") == 3
        assert "content_a1b1c1" in html

    def test_plotly_figure_content(self):
        import plotly.graph_objects as go

        report = self._make_report()
        fig = go.Figure(data=[go.Scatter(x=[1, 2], y=[3, 4])])
        dims = [{"label": "View", "options": ["Chart", "Table"]}]
        content = {
            ("Chart",): fig,
            ("Table",): "<table><tr><td>data</td></tr></table>",
        }
        report.add_cascading_content("Test", dims, content)
        html = report.html_report
        assert "plotly" in html.lower() or "scatter" in html.lower()
        assert "<table>" in html

    def test_first_item_visible(self):
        report = self._make_report()
        dims = [{"label": "X", "options": ["A", "B", "C"]}]
        content = {("A",): "AAA", ("B",): "BBB", ("C",): "CCC"}
        report.add_cascading_content("Test", dims, content)
        html = report.html_report
        # First div should be visible
        first_div_idx = html.find("AAA")
        assert first_div_idx > 0
        # Check that the first wrap div has display:block
        block_before_first = html[:first_div_idx].rfind("display:")
        assert "block" in html[block_before_first : block_before_first + 20]

    def test_cascade_mode(self):
        report = self._make_report()
        dims = [
            {"label": "Country", "options": ["US", "EU"]},
            {"label": "City", "options": ["NYC", "LA", "Berlin", "Paris"]},
        ]
        content = {
            ("US", "NYC"): "US-NYC",
            ("US", "LA"): "US-LA",
            ("EU", "Berlin"): "EU-Berlin",
            ("EU", "Paris"): "EU-Paris",
        }
        report.add_cascading_content("Test", dims, content, cascade=True)
        html = report.html_report
        # Should contain JSON options mapping for cascading
        assert "US" in html
        assert "Berlin" in html

    def test_special_chars_in_options(self):
        report = self._make_report()
        dims = [{"label": "Item", "options": ["foo-bar", "hello world", "a.b.c"]}]
        content = {
            ("foo-bar",): "content1",
            ("hello world",): "content2",
            ("a.b.c",): "content3",
        }
        report.add_cascading_content("Test", dims, content)
        # Should not crash — special chars sanitized in IDs
        html = report.html_report
        assert "content1" in html
        assert "content2" in html
        assert "content3" in html

    def test_js_contains_update_function(self):
        report = self._make_report()
        dims = [{"label": "X", "options": ["A", "B"]}]
        content = {("A",): "a", ("B",): "b"}
        report.add_cascading_content("Test", dims, content)
        html = report.html_report
        assert "function update_" in html or "function upd_" in html
        assert "DOMContentLoaded" in html
        assert "resize" in html


class TestDataframeThresholds:
    """Tests for add_dataframe with threshold-based conditional formatting."""

    def _make_report(self):
        from scomp_link.utils.report_html import ScompLinkHTMLReport

        return ScompLinkHTMLReport("Test Report")

    def test_no_thresholds_unchanged(self):
        report = self._make_report()
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        report.add_dataframe(df, "Test Table")
        html = report.html_report
        assert "scomp-table" in html
        assert "download_table_as_csv" in html

    def test_thresholds_green_orange_red(self):
        report = self._make_report()
        df = pd.DataFrame({"metric": [0.05, 0.15, 0.30]})
        thresholds = {"metric": (0.10, 0.25, False)}  # lower is better
        report.add_dataframe(df, "Metrics", thresholds=thresholds)
        html = report.html_report
        assert "rgba(52,211,153,0.3)" in html  # green for 0.05
        assert "rgba(251,146,60,0.3)" in html  # orange for 0.15
        assert "rgba(239,68,68,0.3)" in html  # red for 0.30

    def test_thresholds_higher_is_better(self):
        report = self._make_report()
        df = pd.DataFrame({"accuracy": [0.95, 0.70, 0.30]})
        thresholds = {"accuracy": (0.8, 0.5, True)}  # higher is better
        report.add_dataframe(df, "Acc", thresholds=thresholds)
        html = report.html_report
        assert "rgba(52,211,153,0.3)" in html  # green for 0.95
        assert "rgba(251,146,60,0.3)" in html  # orange for 0.70
        assert "rgba(239,68,68,0.3)" in html  # red for 0.30

    def test_thresholds_nan_grey(self):
        report = self._make_report()
        df = pd.DataFrame({"val": [0.5, float("nan"), 0.1]})
        thresholds = {"val": (0.3, 0.6, False)}
        report.add_dataframe(df, "NaN Test", thresholds=thresholds)
        html = report.html_report
        assert "rgba(100,100,100,0.2)" in html  # grey for NaN

    def test_thresholds_polars_input(self):
        try:
            import polars as pl
        except ImportError:
            pytest.skip("polars not installed")
        report = self._make_report()
        df = pl.DataFrame({"score": [0.9, 0.5, 0.2]})
        thresholds = {"score": (0.7, 0.4, True)}
        report.add_dataframe(df, "Polars Test", thresholds=thresholds)
        html = report.html_report
        assert "rgba(52,211,153,0.3)" in html

    def test_thresholds_multiple_columns(self):
        report = self._make_report()
        df = pd.DataFrame({"error": [0.01, 0.2, 0.5], "r2": [0.99, 0.6, 0.3]})
        thresholds = {
            "error": (0.1, 0.3, False),
            "r2": (0.8, 0.5, True),
        }
        report.add_dataframe(df, "Multi", thresholds=thresholds)
        html = report.html_report
        # Both green cells should be present
        assert html.count("rgba(52,211,153,0.3)") >= 2


class TestPlotlyUtils:
    """Tests for new plotly utility functions."""

    def test_fill_timeslots_pad(self):
        from scomp_link.utils.plotly_utils import fill_timeslots

        result = fill_timeslots([1, 2, 3], 5)
        assert len(result) == 5
        assert result[0] == 1.0
        assert result[3] == 0.0
        assert result[4] == 0.0

    def test_fill_timeslots_truncate(self):
        from scomp_link.utils.plotly_utils import fill_timeslots

        result = fill_timeslots([1, 2, 3, 4, 5], 3)
        assert len(result) == 3
        assert list(result) == [1.0, 2.0, 3.0]

    def test_fill_timeslots_exact(self):
        from scomp_link.utils.plotly_utils import fill_timeslots

        result = fill_timeslots([1, 2, 3], 3)
        assert len(result) == 3
        assert list(result) == [1.0, 2.0, 3.0]

    def test_fill_timeslots_custom_fill(self):
        from scomp_link.utils.plotly_utils import fill_timeslots

        result = fill_timeslots([1], 3, fill_value=-1.0)
        assert list(result) == [1.0, -1.0, -1.0]

    def test_normalize_to_index_basic(self):
        from scomp_link.utils.plotly_utils import normalize_to_index

        result = normalize_to_index([2, 4, 6], baseline=100)
        assert abs(result.mean() - 100) < 1e-10
        assert abs(result[0] - 50) < 1e-10
        assert abs(result[2] - 150) < 1e-10

    def test_normalize_to_index_zeros(self):
        from scomp_link.utils.plotly_utils import normalize_to_index

        result = normalize_to_index([0, 0, 0], baseline=100)
        assert all(v == 0 for v in result)

    def test_normalize_to_index_custom_baseline(self):
        from scomp_link.utils.plotly_utils import normalize_to_index

        result = normalize_to_index([1, 1, 1], baseline=50)
        assert abs(result.mean() - 50) < 1e-10

    def test_index_chart_simple(self):
        from scomp_link.utils.plotly_utils import index_chart

        fig = index_chart(
            {"A": [1, 2, 3, 4], "B": [4, 3, 2, 1]},
            x_labels=["Q1", "Q2", "Q3", "Q4"],
            title="Test Index",
        )
        assert hasattr(fig, "data")
        assert len(fig.data) == 2  # one trace per series
        assert fig.layout.title.text == "Test Index — A"

    def test_index_chart_paired(self):
        from scomp_link.utils.plotly_utils import index_chart

        fig = index_chart(
            {
                "X": {"solid": [1, 2], "dashed": [2, 1]},
                "Y": {"solid": [3, 4], "dashed": [4, 3]},
            },
            x_labels=["a", "b"],
            title="Paired",
        )
        assert len(fig.data) == 4  # 2 groups × 2 traces
        # Check first group visible, second hidden
        assert fig.data[0].visible is True or fig.data[0].visible is None
        assert fig.data[2].visible is False

    def test_index_chart_has_buttons(self):
        from scomp_link.utils.plotly_utils import index_chart

        fig = index_chart(
            {"A": [1, 2], "B": [3, 4]},
            x_labels=["x", "y"],
            title="Test",
        )
        assert fig.layout.updatemenus is not None
        assert len(fig.layout.updatemenus) == 1
        # buttons: one per series + "All"
        assert len(fig.layout.updatemenus[0].buttons) == 3

    def test_stacked_area_comparison(self):
        from scomp_link.utils.plotly_utils import stacked_area_comparison

        fig = stacked_area_comparison(
            data_left={"A": [30, 40], "B": [70, 60]},
            data_right={"A": [50, 50], "B": [50, 50]},
            categories=["A", "B"],
            x_labels=["t1", "t2"],
            title="Compare",
            subplot_titles=("Left", "Right"),
        )
        assert hasattr(fig, "data")
        # 2 categories × 2 subplots = 4 traces
        assert len(fig.data) == 4

    def test_stacked_area_normalizes_to_100(self):
        from scomp_link.utils.plotly_utils import stacked_area_comparison

        fig = stacked_area_comparison(
            data_left={"A": [10, 20], "B": [30, 40]},
            data_right={"A": [5, 5], "B": [5, 5]},
            categories=["A", "B"],
            x_labels=["t1", "t2"],
            title="Norm Test",
        )
        # Left subplot: first slot A=25%, B=75%
        assert abs(fig.data[0].y[0] - 25.0) < 1e-10
        assert abs(fig.data[1].y[0] - 75.0) < 1e-10


class TestDeprecation:
    """Tests for deprecated select_plotly and add_many_plots_with_selection_box_to_report."""

    def _make_report(self):
        from scomp_link.utils.report_html import ScompLinkHTMLReport

        return ScompLinkHTMLReport("Test Report")

    def test_select_plotly_emits_warning(self):
        import plotly.graph_objects as go

        report = self._make_report()
        figs = {"fig1": go.Figure(), "fig2": go.Figure()}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            report.select_plotly(figs, "Test")
            assert len(w) >= 1
            assert issubclass(w[0].category, DeprecationWarning)
            assert "add_cascading_content" in str(w[0].message)

    def test_select_plotly_still_works(self):
        import plotly.graph_objects as go

        report = self._make_report()
        figs = {"A": go.Figure(), "B": go.Figure()}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            report.select_plotly(figs, "Test")
        assert len(report.html_report) > 0
        assert "<select" in report.html_report

    def test_add_many_plots_emits_warning(self):
        import plotly.graph_objects as go

        report = self._make_report()
        figs = {"fig1": go.Figure(), "fig2": go.Figure()}
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            report.add_many_plots_with_selection_box_to_report(figs, "Test")
            assert len(w) >= 1
            assert issubclass(w[0].category, DeprecationWarning)

    def test_select_plotly_tuple_keys(self):
        import plotly.graph_objects as go

        report = self._make_report()
        figs = {
            ("US", "2024"): go.Figure(),
            ("US", "2025"): go.Figure(),
            ("EU", "2024"): go.Figure(),
            ("EU", "2025"): go.Figure(),
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            report.select_plotly(figs, "Multi", labels=["Region", "Year"])
        html = report.html_report
        assert len(html) > 0
        assert "<select" in html


# ═══════════════════════════════════════════════════════════════════
# Tests for advanced report components
# ═══════════════════════════════════════════════════════════════════


class TestKPICards:
    """Tests for add_kpi_cards."""

    def _make_report(self):
        from scomp_link.utils.report_html import ScompLinkHTMLReport
        return ScompLinkHTMLReport("Test Report")

    def test_basic_kpi_cards(self):
        report = self._make_report()
        report.add_kpi_cards({
            "Accuracy": {"value": "94.2%", "trend": "+1.3%", "status": "good"},
            "RMSE": {"value": "0.087", "status": "warning"},
            "Drift": {"value": "0.31", "status": "critical"},
        })
        html = report.html_report
        assert "94.2%" in html
        assert "0.087" in html
        assert "+1.3%" in html
        assert "grid" in html

    def test_kpi_simple_values(self):
        report = self._make_report()
        report.add_kpi_cards({"Count": "1234", "Score": 0.95})
        html = report.html_report
        assert "1234" in html
        assert "0.95" in html

    def test_kpi_custom_cols(self):
        report = self._make_report()
        report.add_kpi_cards({"A": "1", "B": "2", "C": "3", "D": "4"}, cols=4)
        html = report.html_report
        assert "repeat(4,1fr)" in html

    def test_kpi_subtitle(self):
        report = self._make_report()
        report.add_kpi_cards({"Samples": {"value": "12,450", "subtitle": "last 24h"}})
        html = report.html_report
        assert "last 24h" in html

    def test_kpi_trend_arrows(self):
        report = self._make_report()
        report.add_kpi_cards({
            "Up": {"value": "10", "trend": "+5", "status": "good"},
            "Down": {"value": "10", "trend": "-3", "status": "critical"},
        })
        html = report.html_report
        assert "↑" in html
        assert "↓" in html


class TestPlotlyGrid:
    """Tests for add_plotly_grid."""

    def _make_report(self):
        from scomp_link.utils.report_html import ScompLinkHTMLReport
        return ScompLinkHTMLReport("Test Report")

    def test_basic_grid(self):
        import plotly.graph_objects as go
        report = self._make_report()
        figs = [go.Figure() for _ in range(4)]
        report.add_plotly_grid(figs, cols=2)
        html = report.html_report
        assert "grid-template-columns" in html
        assert html.count("plotly") >= 4

    def test_grid_with_titles(self):
        import plotly.graph_objects as go
        report = self._make_report()
        report.add_plotly_grid([go.Figure(), go.Figure()], cols=2, titles=["Chart A", "Chart B"])
        html = report.html_report
        assert "Chart A" in html
        assert "Chart B" in html

    def test_grid_responsive(self):
        import plotly.graph_objects as go
        report = self._make_report()
        report.add_plotly_grid([go.Figure()], cols=3)
        html = report.html_report
        assert "768px" in html  # responsive breakpoint


class TestTabs:
    """Tests for add_tabs."""

    def _make_report(self):
        from scomp_link.utils.report_html import ScompLinkHTMLReport
        return ScompLinkHTMLReport("Test Report")

    def test_basic_tabs(self):
        report = self._make_report()
        report.add_tabs({"Tab 1": "<p>Content 1</p>", "Tab 2": "<p>Content 2</p>"})
        html = report.html_report
        assert "Content 1" in html
        assert "Content 2" in html
        assert "switchTab_" in html

    def test_tabs_with_plotly(self):
        import plotly.graph_objects as go
        report = self._make_report()
        report.add_tabs({"Chart": go.Figure(data=[go.Scatter(x=[1], y=[2])])})
        html = report.html_report
        assert "plotly" in html.lower()

    def test_tabs_with_dataframe(self):
        report = self._make_report()
        report.add_tabs({"Data": pd.DataFrame({"x": [1, 2, 3]})})
        html = report.html_report
        assert "<table" in html

    def test_tabs_first_visible(self):
        report = self._make_report()
        report.add_tabs({"A": "aaa", "B": "bbb", "C": "ccc"})
        html = report.html_report
        # First panel should be display:block
        first_panel = html.find("aaa")
        block_before = html[:first_panel].rfind("display:")
        assert "block" in html[block_before:block_before + 20]

    def test_tabs_with_title(self):
        report = self._make_report()
        report.add_tabs({"X": "content"}, title="My Tabs")
        html = report.html_report
        assert "My Tabs" in html

    def test_tabs_resize_on_switch(self):
        report = self._make_report()
        report.add_tabs({"A": "a", "B": "b"})
        html = report.html_report
        assert "Plotly.Plots.resize" in html


class TestComparisonTable:
    """Tests for add_comparison_table."""

    def _make_report(self):
        from scomp_link.utils.report_html import ScompLinkHTMLReport
        return ScompLinkHTMLReport("Test Report")

    def test_basic_comparison(self):
        report = self._make_report()
        df = pd.DataFrame({"metric": ["acc"], "v1": [0.90], "v2": [0.95]})
        report.add_comparison_table(df, "v1", ["v2"], metric_col="metric")
        html = report.html_report
        assert "baseline" in html
        assert "↑" in html  # improvement

    def test_higher_is_better_false(self):
        report = self._make_report()
        df = pd.DataFrame({"metric": ["rmse"], "v1": [0.12], "v2": [0.09]})
        report.add_comparison_table(df, "v1", ["v2"], metric_col="metric",
                                    higher_is_better={"rmse": False})
        html = report.html_report
        assert "0f9d58" in html  # green color = improvement

    def test_multiple_compare_cols(self):
        report = self._make_report()
        df = pd.DataFrame({"m": ["x"], "a": [1.0], "b": [2.0], "c": [0.5]})
        report.add_comparison_table(df, "a", ["b", "c"], metric_col="m")
        html = report.html_report
        assert "vs baseline" in html

    def test_polars_input(self):
        try:
            import polars as pl
        except ImportError:
            pytest.skip("polars not installed")
        report = self._make_report()
        df = pl.DataFrame({"metric": ["acc"], "v1": [0.9], "v2": [0.95]})
        report.add_comparison_table(df, "v1", ["v2"], metric_col="metric")
        assert len(report.html_report) > 0


class TestSummaryStats:
    """Tests for add_summary_stats."""

    def _make_report(self):
        from scomp_link.utils.report_html import ScompLinkHTMLReport
        return ScompLinkHTMLReport("Test Report")

    def test_basic_summary(self):
        report = self._make_report()
        df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", None]})
        report.add_summary_stats(df)
        html = report.html_report
        assert "Data Summary" in html
        assert "Missing %" in html
        assert "3 rows" in html

    def test_custom_title(self):
        report = self._make_report()
        report.add_summary_stats(pd.DataFrame({"x": [1]}), title="My Stats")
        assert "My Stats" in report.html_report

    def test_missing_percentage_shown(self):
        report = self._make_report()
        df = pd.DataFrame({"col": [1, None, None, 4]})
        report.add_summary_stats(df)
        html = report.html_report
        assert "50.0%" in html  # 2 out of 4 missing

    def test_type_badges(self):
        report = self._make_report()
        df = pd.DataFrame({"num": [1.0], "text": ["hello"], "flag": [True]})
        report.add_summary_stats(df)
        html = report.html_report
        assert "float" in html
        assert "object" in html
        assert "bool" in html


class TestDarkModeToggle:
    """Tests for add_dark_mode_toggle."""

    def _make_report(self):
        from scomp_link.utils.report_html import ScompLinkHTMLReport
        return ScompLinkHTMLReport("Test Report")

    def test_toggle_added(self):
        report = self._make_report()
        report.add_dark_mode_toggle()
        html = report.html_report
        assert "toggleDarkMode_" in html
        assert "🌙" in html

    def test_toggle_has_dark_vars(self):
        report = self._make_report()
        report.add_dark_mode_toggle()
        html = report.html_report
        assert "#0f172a" in html  # dark background
        assert "--bg" in html

    def test_toggle_fixed_position(self):
        report = self._make_report()
        report.add_dark_mode_toggle()
        assert "position:fixed" in report.html_report
