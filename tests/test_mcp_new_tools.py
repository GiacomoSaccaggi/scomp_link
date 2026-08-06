# -*- coding: utf-8 -*-
"""Tests for new MCP report tools (Tasks 0-6 of MCP integration plan)."""

import json

import pytest


@pytest.fixture(autouse=True)
def clear_reports():
    """Clear the report store before and after each test."""
    from scomp_link.mcp_server import _reports

    _reports.clear()
    yield
    _reports.clear()


@pytest.fixture()
def report_id():
    """Create a report and return its ID."""
    from scomp_link.mcp_server import report_add_section, report_create

    r = json.loads(report_create("Test Report"))
    rid = r["report_id"]
    report_add_section(rid, "Test Section")
    return rid


# ═══════════════════════════════════════════════════════════════════
# Task 0+1: _build_plotly_figure helper + report_add_chart new types
# ═══════════════════════════════════════════════════════════════════


class TestBuildPlotlyFigure:
    """Tests for the _build_plotly_figure helper."""

    def test_histogram(self):
        from scomp_link.mcp_server import _build_plotly_figure

        fig = _build_plotly_figure("histogram", {"values": [1, 2, 3, 4, 5], "name": "test"}, "Title")
        assert hasattr(fig, "data")

    def test_barchart(self):
        from scomp_link.mcp_server import _build_plotly_figure

        fig = _build_plotly_figure("barchart", {"categories": ["A", "B"], "values": [[10, 20]]}, "Title")
        assert hasattr(fig, "data")

    def test_linechart(self):
        from scomp_link.mcp_server import _build_plotly_figure

        fig = _build_plotly_figure("linechart", {"dates": ["2024-01-01", "2024-02-01"], "lines": [[100, 120]]}, "Title")
        assert hasattr(fig, "data")

    def test_area_chart(self):
        from scomp_link.mcp_server import _build_plotly_figure

        fig = _build_plotly_figure(
            "area_chart", {"dates": ["2024-01-01", "2024-02-01"], "lines": [[100, 120]]}, "Title"
        )
        assert hasattr(fig, "data")

    def test_index_chart_simple(self):
        from scomp_link.mcp_server import _build_plotly_figure

        fig = _build_plotly_figure(
            "index_chart", {"series_dict": {"A": [100, 110, 105]}, "x_labels": ["Q1", "Q2", "Q3"]}, "Index"
        )
        assert hasattr(fig, "data")

    def test_index_chart_paired(self):
        from scomp_link.mcp_server import _build_plotly_figure

        fig = _build_plotly_figure(
            "index_chart",
            {"series_dict": {"A": {"solid": [100, 110], "dashed": [95, 105]}}, "x_labels": ["Q1", "Q2"]},
            "Paired",
        )
        assert hasattr(fig, "data")

    def test_index_chart_custom_baseline(self):
        from scomp_link.mcp_server import _build_plotly_figure

        fig = _build_plotly_figure(
            "index_chart", {"series_dict": {"A": [50, 60]}, "x_labels": ["X", "Y"], "baseline": 50.0}, "Base50"
        )
        assert hasattr(fig, "data")

    def test_stacked_area_comparison(self):
        from scomp_link.mcp_server import _build_plotly_figure

        fig = _build_plotly_figure(
            "stacked_area_comparison",
            {
                "data_left": {"Cat A": [30, 40], "Cat B": [70, 60]},
                "data_right": {"Cat A": [20, 30], "Cat B": [80, 70]},
                "categories": ["Cat A", "Cat B"],
                "x_labels": ["2024", "2025"],
            },
            "Compare",
        )
        assert hasattr(fig, "data")

    def test_stacked_area_comparison_with_subplot_titles(self):
        from scomp_link.mcp_server import _build_plotly_figure

        fig = _build_plotly_figure(
            "stacked_area_comparison",
            {
                "data_left": {"A": [50, 50]},
                "data_right": {"A": [60, 40]},
                "categories": ["A"],
                "x_labels": ["X", "Y"],
                "subplot_titles": ["Left Panel", "Right Panel"],
            },
            "Custom Titles",
        )
        assert hasattr(fig, "data")

    def test_unknown_type_raises(self):
        from scomp_link.mcp_server import _build_plotly_figure

        with pytest.raises(ValueError, match="Unknown plotly chart_type"):
            _build_plotly_figure("nonexistent", {}, "Title")


class TestReportAddChartNewTypes:
    """Tests for report_add_chart with the new plotly types."""

    def test_index_chart(self, report_id):
        from scomp_link.mcp_server import report_add_chart

        res = json.loads(
            report_add_chart(
                report_id,
                "plotly",
                "index_chart",
                json.dumps({"series_dict": {"A": [100, 110]}, "x_labels": ["Jan", "Feb"]}),
                "Index",
            )
        )
        assert res["status"] == "chart_added"
        assert res["chart_type"] == "index_chart"

    def test_stacked_area_comparison(self, report_id):
        from scomp_link.mcp_server import report_add_chart

        res = json.loads(
            report_add_chart(
                report_id,
                "plotly",
                "stacked_area_comparison",
                json.dumps(
                    {
                        "data_left": {"A": [30, 40]},
                        "data_right": {"A": [20, 30]},
                        "categories": ["A"],
                        "x_labels": ["X", "Y"],
                    }
                ),
                "Compare",
            )
        )
        assert res["status"] == "chart_added"

    def test_existing_histogram_still_works(self, report_id):
        from scomp_link.mcp_server import report_add_chart

        res = json.loads(report_add_chart(report_id, "plotly", "histogram", json.dumps({"values": [1, 2, 3]}), "Hist"))
        assert res["status"] == "chart_added"

    def test_invalid_type_error(self, report_id):
        from scomp_link.mcp_server import report_add_chart

        res = json.loads(report_add_chart(report_id, "plotly", "fake_chart", "{}", "Nope"))
        assert "error" in res

    def test_missing_keys_error(self, report_id):
        from scomp_link.mcp_server import report_add_chart

        res = json.loads(report_add_chart(report_id, "plotly", "index_chart", json.dumps({"wrong_key": []}), "Bad"))
        assert "error" in res

    def test_bad_report_id(self):
        from scomp_link.mcp_server import report_add_chart

        res = json.loads(report_add_chart("nonexistent", "plotly", "histogram", json.dumps({"values": [1]}), "X"))
        assert "error" in res


# ═══════════════════════════════════════════════════════════════════
# Task 2: report_add_kpi_cards
# ═══════════════════════════════════════════════════════════════════


class TestReportAddKPICards:
    def test_basic(self, report_id):
        from scomp_link.mcp_server import report_add_kpi_cards

        res = json.loads(
            report_add_kpi_cards(
                report_id, json.dumps({"RMSE": {"value": 0.81, "status": "good"}, "R2": {"value": 0.92}})
            )
        )
        assert res["status"] == "kpi_cards_added"
        assert res["n_metrics"] == 2

    def test_scalar_values(self, report_id):
        from scomp_link.mcp_server import report_add_kpi_cards

        res = json.loads(report_add_kpi_cards(report_id, json.dumps({"Count": 1500, "Label": "OK"})))
        assert res["status"] == "kpi_cards_added"
        assert res["n_metrics"] == 2

    def test_full_config(self, report_id):
        from scomp_link.mcp_server import report_add_kpi_cards

        res = json.loads(
            report_add_kpi_cards(
                report_id,
                json.dumps({"Metric": {"value": "42", "trend": "+5%", "status": "warning", "subtitle": "last 7d"}}),
                cols=4,
            )
        )
        assert res["status"] == "kpi_cards_added"

    def test_invalid_json(self, report_id):
        from scomp_link.mcp_server import report_add_kpi_cards

        res = json.loads(report_add_kpi_cards(report_id, "not valid json {{{"))
        assert "error" in res

    def test_bad_report_id(self):
        from scomp_link.mcp_server import report_add_kpi_cards

        res = json.loads(report_add_kpi_cards("nonexistent", "{}"))
        assert "error" in res


# ═══════════════════════════════════════════════════════════════════
# Task 3: report_add_tabs
# ═══════════════════════════════════════════════════════════════════


class TestReportAddTabs:
    def test_html_tab(self, report_id):
        from scomp_link.mcp_server import report_add_tabs

        res = json.loads(report_add_tabs(report_id, json.dumps({"Info": {"type": "html", "content": "<p>Hello</p>"}})))
        assert res["status"] == "tabs_added"
        assert res["n_tabs"] == 1

    def test_chart_tab(self, report_id):
        from scomp_link.mcp_server import report_add_tabs

        res = json.loads(
            report_add_tabs(
                report_id,
                json.dumps(
                    {
                        "Chart": {
                            "type": "chart",
                            "engine": "plotly",
                            "chart_type": "histogram",
                            "data": {"values": [1, 2, 3, 4, 5]},
                        }
                    }
                ),
            )
        )
        assert res["status"] == "tabs_added"

    def test_table_tab(self, report_id):
        from scomp_link.mcp_server import report_add_tabs

        res = json.loads(
            report_add_tabs(report_id, json.dumps({"Data": {"type": "table", "data": [{"a": 1}, {"a": 2}]}}))
        )
        assert res["status"] == "tabs_added"

    def test_all_three_types(self, report_id):
        from scomp_link.mcp_server import report_add_tabs

        tabs = {
            "Overview": {"type": "html", "content": "<b>Summary</b>"},
            "Chart": {
                "type": "chart",
                "engine": "plotly",
                "chart_type": "barchart",
                "data": {"categories": ["A"], "values": [[5]]},
            },
            "Table": {"type": "table", "data": [{"x": 1}]},
        }
        res = json.loads(report_add_tabs(report_id, json.dumps(tabs), "Results"))
        assert res["status"] == "tabs_added"
        assert res["n_tabs"] == 3

    def test_unsupported_engine_in_tab(self, report_id):
        from scomp_link.mcp_server import report_add_tabs

        res = json.loads(
            report_add_tabs(
                report_id,
                json.dumps({"X": {"type": "chart", "engine": "rawgraphs", "chart_type": "treemap", "data": {}}}),
            )
        )
        assert "error" in res

    def test_unknown_tab_type(self, report_id):
        from scomp_link.mcp_server import report_add_tabs

        res = json.loads(report_add_tabs(report_id, json.dumps({"Bad": {"type": "unknown"}})))
        assert "error" in res

    def test_invalid_json(self, report_id):
        from scomp_link.mcp_server import report_add_tabs

        res = json.loads(report_add_tabs(report_id, "{{invalid"))
        assert "error" in res

    def test_bad_report_id(self):
        from scomp_link.mcp_server import report_add_tabs

        res = json.loads(report_add_tabs("nonexistent", "{}"))
        assert "error" in res


# ═══════════════════════════════════════════════════════════════════
# Task 4: report_add_comparison_table
# ═══════════════════════════════════════════════════════════════════


class TestReportAddComparisonTable:
    def test_basic(self, report_id):
        from scomp_link.mcp_server import report_add_comparison_table

        data = [{"metric": "acc", "v1": 0.9, "v2": 0.95}, {"metric": "rmse", "v1": 0.1, "v2": 0.08}]
        res = json.loads(
            report_add_comparison_table(
                report_id, json.dumps(data), baseline_col="v1", compare_cols=json.dumps(["v2"]), metric_col="metric"
            )
        )
        assert res["status"] == "comparison_table_added"

    def test_with_higher_is_better(self, report_id):
        from scomp_link.mcp_server import report_add_comparison_table

        data = [{"metric": "acc", "v1": 0.9, "v2": 0.95}, {"metric": "rmse", "v1": 0.1, "v2": 0.08}]
        res = json.loads(
            report_add_comparison_table(
                report_id,
                json.dumps(data),
                baseline_col="v1",
                compare_cols=json.dumps(["v2"]),
                metric_col="metric",
                higher_is_better=json.dumps({"acc": True, "rmse": False}),
            )
        )
        assert res["status"] == "comparison_table_added"

    def test_no_higher_is_better(self, report_id):
        from scomp_link.mcp_server import report_add_comparison_table

        data = [{"v1": 10, "v2": 12, "v3": 8}]
        res = json.loads(
            report_add_comparison_table(
                report_id, json.dumps(data), baseline_col="v1", compare_cols=json.dumps(["v2", "v3"])
            )
        )
        assert res["status"] == "comparison_table_added"

    def test_invalid_compare_cols_json(self, report_id):
        from scomp_link.mcp_server import report_add_comparison_table

        res = json.loads(
            report_add_comparison_table(report_id, json.dumps([{"v1": 1}]), baseline_col="v1", compare_cols="not json")
        )
        assert "error" in res

    def test_bad_report_id(self):
        from scomp_link.mcp_server import report_add_comparison_table

        res = json.loads(report_add_comparison_table("nonexistent", "[]", "v1", '["v2"]'))
        assert "error" in res


# ═══════════════════════════════════════════════════════════════════
# Task 5: report_add_summary_stats
# ═══════════════════════════════════════════════════════════════════


class TestReportAddSummaryStats:
    def test_basic(self, report_id):
        from scomp_link.mcp_server import report_add_summary_stats

        data = [{"age": 25, "name": "Alice"}, {"age": 30, "name": "Bob"}, {"age": 35, "name": "Charlie"}]
        res = json.loads(report_add_summary_stats(report_id, json.dumps(data)))
        assert res["status"] == "summary_stats_added"
        assert res["n_columns"] == 2

    def test_with_missing_values(self, report_id):
        from scomp_link.mcp_server import report_add_summary_stats

        data = [{"a": 1, "b": None}, {"a": None, "b": 2}]
        res = json.loads(report_add_summary_stats(report_id, json.dumps(data), title="Sparse Data"))
        assert res["status"] == "summary_stats_added"

    def test_empty_array(self, report_id):
        from scomp_link.mcp_server import report_add_summary_stats

        res = json.loads(report_add_summary_stats(report_id, "[]"))
        # Empty DataFrame — should either work with 0 columns or error gracefully
        assert "status" in res or "error" in res

    def test_invalid_json(self, report_id):
        from scomp_link.mcp_server import report_add_summary_stats

        res = json.loads(report_add_summary_stats(report_id, "not json"))
        assert "error" in res

    def test_bad_report_id(self):
        from scomp_link.mcp_server import report_add_summary_stats

        res = json.loads(report_add_summary_stats("nonexistent", "[]"))
        assert "error" in res


# ═══════════════════════════════════════════════════════════════════
# Task 6: report_add_dark_mode_toggle
# ═══════════════════════════════════════════════════════════════════


class TestReportAddDarkModeToggle:
    def test_basic(self, report_id):
        from scomp_link.mcp_server import report_add_dark_mode_toggle

        res = json.loads(report_add_dark_mode_toggle(report_id))
        assert res["status"] == "dark_mode_toggle_added"

    def test_before_any_section(self):
        """Toggle can be added even right after report_create (no section needed)."""
        from scomp_link.mcp_server import report_add_dark_mode_toggle, report_create

        r = json.loads(report_create("Bare Report"))
        res = json.loads(report_add_dark_mode_toggle(r["report_id"]))
        assert res["status"] == "dark_mode_toggle_added"

    def test_bad_report_id(self):
        from scomp_link.mcp_server import report_add_dark_mode_toggle

        res = json.loads(report_add_dark_mode_toggle("nonexistent"))
        assert "error" in res


# ═══════════════════════════════════════════════════════════════════
# End-to-end workflow test
# ═══════════════════════════════════════════════════════════════════


class TestEndToEndWorkflow:
    """Integration test: full report-building session with all new tools."""

    def test_full_workflow(self, tmp_path):
        from scomp_link.mcp_server import (
            _reports,
            report_add_chart,
            report_add_comparison_table,
            report_add_dark_mode_toggle,
            report_add_kpi_cards,
            report_add_section,
            report_add_summary_stats,
            report_add_tabs,
            report_create,
            report_save,
        )

        # 1. Create
        r = json.loads(report_create("E2E Test Report"))
        rid = r["report_id"]
        assert rid in _reports

        # 2. Dark mode toggle (before any section)
        res = json.loads(report_add_dark_mode_toggle(rid))
        assert res["status"] == "dark_mode_toggle_added"

        # 3. Section + KPI cards
        report_add_section(rid, "Summary")
        res = json.loads(
            report_add_kpi_cards(
                rid,
                json.dumps(
                    {
                        "Accuracy": {"value": "94%", "trend": "+2%", "status": "good"},
                        "Latency": {"value": "120ms", "status": "warning"},
                    }
                ),
            )
        )
        assert res["n_metrics"] == 2

        # 4. Section + Tabs
        report_add_section(rid, "Details")
        res = json.loads(
            report_add_tabs(
                rid,
                json.dumps(
                    {
                        "Overview": {"type": "html", "content": "<p>All good</p>"},
                        "Data": {"type": "table", "data": [{"x": 1, "y": 2}]},
                    }
                ),
                "Results",
            )
        )
        assert res["n_tabs"] == 2

        # 5. Comparison table
        res = json.loads(
            report_add_comparison_table(
                rid,
                json.dumps([{"metric": "f1", "v1": 0.85, "v2": 0.90}]),
                baseline_col="v1",
                compare_cols=json.dumps(["v2"]),
                metric_col="metric",
            )
        )
        assert res["status"] == "comparison_table_added"

        # 6. Summary stats
        res = json.loads(
            report_add_summary_stats(
                rid,
                json.dumps(
                    [
                        {"a": 1, "b": "x"},
                        {"a": 2, "b": "y"},
                        {"a": 3, "b": "z"},
                    ]
                ),
            )
        )
        assert res["n_columns"] == 2

        # 7. Chart (new type)
        res = json.loads(
            report_add_chart(
                rid,
                "plotly",
                "index_chart",
                json.dumps({"series_dict": {"A": [100, 110, 105]}, "x_labels": ["Q1", "Q2", "Q3"]}),
                "Index",
            )
        )
        assert res["status"] == "chart_added"

        # 8. Save
        out = tmp_path / "e2e_report.html"
        res = json.loads(report_save(rid, str(out)))
        assert res["status"] == "saved"
        assert out.exists()
        assert out.stat().st_size > 1000  # Non-trivial HTML
        assert rid not in _reports  # Session cleaned up
