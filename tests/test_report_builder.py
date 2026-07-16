# -*- coding: utf-8 -*-
"""Tests for Report Builder MCP tools and config system."""

import json
import os
import tempfile

import pytest

# ═══════════════════════════════════════════════════════════════════
# Config System Tests
# ═══════════════════════════════════════════════════════════════════


class TestConfig:
    """Tests for scomp_link.config module."""

    def test_hardcoded_defaults(self):
        from scomp_link.config import _HARDCODED_DEFAULTS

        assert _HARDCODED_DEFAULTS["report"]["main_color"] == "#6E37FA"
        assert _HARDCODED_DEFAULTS["report"]["footer_html"] is None
        assert _HARDCODED_DEFAULTS["report"]["font_family"] == "Baloo 2"

    def test_load_config_returns_defaults(self):
        from scomp_link.config import load_config
        from pathlib import Path
        import unittest.mock as mock

        # Mock config paths to ensure only hardcoded defaults are used
        with mock.patch("scomp_link.config._GLOBAL_CONFIG_PATH", Path("/nonexistent/cfg.yaml")):
            with mock.patch("scomp_link.config._LOCAL_CONFIG_PATH", Path("/nonexistent/.scomp-link.yaml")):
                cfg = load_config()
                assert "report" in cfg
                assert cfg["report"]["main_color"] == "#6E37FA"
                assert cfg["report"]["author"] == "scomp-link toolkit"

    def test_get_report_defaults(self):
        from scomp_link.config import get_report_defaults
        from pathlib import Path
        import unittest.mock as mock

        with mock.patch("scomp_link.config._GLOBAL_CONFIG_PATH", Path("/nonexistent/cfg.yaml")):
            with mock.patch("scomp_link.config._LOCAL_CONFIG_PATH", Path("/nonexistent/.scomp-link.yaml")):
                defaults = get_report_defaults()
                assert defaults["font_family"] == "Baloo 2"
                assert defaults["language"] == "en"
                assert defaults["dark_color"] == "#4614B4"
                assert defaults["footer_html"] is None

    def test_init_config_creates_file(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "config.yaml")
            from scomp_link.config import init_config

            result = init_config(path=path)
            assert os.path.exists(result)
            with open(result) as f:
                content = f.read()
            assert "report:" in content
            assert "main_color" in content
            assert "Pirelli" in content  # Example in comments

    def test_init_config_raises_if_exists(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "config.yaml")
            from scomp_link.config import init_config

            init_config(path=path)
            with pytest.raises(FileExistsError):
                init_config(path=path)

    def test_init_config_force_overwrites(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "config.yaml")
            from scomp_link.config import init_config

            init_config(path=path)
            init_config(path=path, force=True)  # Should not raise

    def test_local_config_overrides_defaults(self):
        """Test that a local .scomp-link.yaml overrides defaults."""
        import unittest.mock as mock

        import yaml
        from pathlib import Path

        from scomp_link.config import _LOCAL_CONFIG_PATH, load_config

        # Write a local config
        local_cfg = {"report": {"main_color": "#FF0000", "author": "Test Corp"}}
        _LOCAL_CONFIG_PATH.write_text(yaml.dump(local_cfg), encoding="utf-8")

        try:
            # Mock global config to avoid interference from real ~/.scomp-link/config.yaml
            with mock.patch("scomp_link.config._GLOBAL_CONFIG_PATH", Path("/nonexistent/cfg.yaml")):
                cfg = load_config()
                assert cfg["report"]["main_color"] == "#FF0000"
                assert cfg["report"]["author"] == "Test Corp"
                # Non-overridden defaults should remain
                assert cfg["report"]["font_family"] == "Baloo 2"
        finally:
            _LOCAL_CONFIG_PATH.unlink(missing_ok=True)

    def test_deep_merge(self):
        from scomp_link.config import _deep_merge

        base = {"a": {"x": 1, "y": 2}, "b": 3}
        override = {"a": {"y": 99, "z": 100}, "c": 4}
        result = _deep_merge(base, override)
        assert result == {"a": {"x": 1, "y": 99, "z": 100}, "b": 3, "c": 4}

    def test_load_yaml_missing_file(self):
        from pathlib import Path

        from scomp_link.config import _load_yaml

        result = _load_yaml(Path("/nonexistent/path.yaml"))
        assert result == {}


# ═══════════════════════════════════════════════════════════════════
# Footer Parametrization Tests
# ═══════════════════════════════════════════════════════════════════


class TestFooterParametrization:
    """Tests for footer_html parameter in ScompLinkHTMLReport."""

    def test_default_footer(self):
        from scomp_link.utils.report_html import _DEFAULT_FOOTER_CONTENT, ScompLinkHTMLReport

        report = ScompLinkHTMLReport("Test Report")
        assert _DEFAULT_FOOTER_CONTENT in report.footer
        assert "About scomp-link" in report.footer

    def test_custom_footer(self):
        from scomp_link.utils.report_html import _DEFAULT_FOOTER_CONTENT, ScompLinkHTMLReport

        custom = "<footer><strong>Pirelli S.p.A.</strong><br>Confidential.</footer>"
        report = ScompLinkHTMLReport("Test Report", footer_html=custom)
        assert custom in report.footer
        assert _DEFAULT_FOOTER_CONTENT not in report.footer

    def test_js_scripts_always_present(self):
        from scomp_link.utils.report_html import ScompLinkHTMLReport

        custom = "<footer>Custom Footer</footer>"
        report = ScompLinkHTMLReport("Test", footer_html=custom)
        assert "<script>" in report.footer
        assert "download_table_as_csv" in report.footer
        assert "resizeElements" in report.footer

    def test_custom_footer_in_saved_html(self):
        from scomp_link.utils.report_html import ScompLinkHTMLReport

        custom = "<footer><strong>Pirelli</strong></footer>"
        report = ScompLinkHTMLReport("Test", footer_html=custom)
        report.open_section("Section")
        report.add_text("Content")
        report.close_section()

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            report.save_html(path)
            with open(path) as f:
                html = f.read()
            assert "Pirelli" in html
            assert "About scomp-link" not in html
        finally:
            os.unlink(path)


# ═══════════════════════════════════════════════════════════════════
# Report Builder MCP Tools Tests
# ═══════════════════════════════════════════════════════════════════


class TestReportCreate:
    """Tests for report_create MCP tool."""

    def setup_method(self):
        from scomp_link.mcp_server import _reports

        _reports.clear()

    def test_basic_create(self):
        from scomp_link.mcp_server import _reports, report_create

        result = json.loads(report_create("Test Report"))
        assert "report_id" in result
        assert result["title"] == "Test Report"
        assert result["report_id"] in _reports

    def test_create_with_custom_params(self):
        from scomp_link.mcp_server import _reports, report_create

        result = json.loads(
            report_create(
                "Custom Report",
                main_color="#CC0000",
                author="Pirelli Digital",
                footer_html="<footer>Pirelli</footer>",
            )
        )
        rid = result["report_id"]
        assert result["params"]["main_color"] == "#CC0000"
        assert result["params"]["author"] == "Pirelli Digital"
        assert "Pirelli" in _reports[rid].footer

    def test_create_uses_config_defaults(self):
        """report_create with None params should use config defaults."""
        from scomp_link.mcp_server import report_create
        from pathlib import Path
        import unittest.mock as mock

        with mock.patch("scomp_link.config._GLOBAL_CONFIG_PATH", Path("/nonexistent/cfg.yaml")):
            with mock.patch("scomp_link.config._LOCAL_CONFIG_PATH", Path("/nonexistent/.scomp-link.yaml")):
                result = json.loads(report_create("Default Report"))
                assert result["params"]["font_family"] == "Baloo 2"
                assert result["params"]["main_color"] == "#6E37FA"

    def teardown_method(self):
        from scomp_link.mcp_server import _reports

        _reports.clear()


class TestReportAddSection:
    """Tests for report_add_section MCP tool."""

    def setup_method(self):
        from scomp_link.mcp_server import _reports, report_create

        _reports.clear()
        result = json.loads(report_create("Test"))
        self.rid = result["report_id"]

    def test_open_section(self):
        from scomp_link.mcp_server import _reports, report_add_section

        result = json.loads(report_add_section(self.rid, "Overview"))
        assert result["status"] == "section_opened"
        assert _reports[self.rid].section_just_open is True

    def test_auto_close_previous_section(self):
        from scomp_link.mcp_server import _reports, report_add_section

        report_add_section(self.rid, "Section 1")
        report_add_section(self.rid, "Section 2")
        # Should have closed section 1 and opened section 2
        assert _reports[self.rid].section_just_open is True

    def test_invalid_report_id(self):
        from scomp_link.mcp_server import report_add_section

        result = json.loads(report_add_section("nonexistent", "Title"))
        assert "error" in result

    def teardown_method(self):
        from scomp_link.mcp_server import _reports

        _reports.clear()


class TestReportAddText:
    """Tests for report_add_text MCP tool."""

    def setup_method(self):
        from scomp_link.mcp_server import _reports, report_add_section, report_create

        _reports.clear()
        result = json.loads(report_create("Test"))
        self.rid = result["report_id"]
        report_add_section(self.rid, "Section")

    def test_add_paragraph(self):
        from scomp_link.mcp_server import _reports, report_add_text

        result = json.loads(report_add_text(self.rid, "Hello world"))
        assert result["status"] == "text_added"
        assert result["style"] == "paragraph"
        assert "Hello world" in _reports[self.rid].html_report

    def test_add_title(self):
        from scomp_link.mcp_server import _reports, report_add_text

        report_add_text(self.rid, "Big Title", style="title")
        assert "Big Title" in _reports[self.rid].html_report

    def test_add_subtitle(self):
        from scomp_link.mcp_server import _reports, report_add_text

        report_add_text(self.rid, "Subtitle", style="subtitle")
        assert "<h3>Subtitle</h3>" in _reports[self.rid].html_report

    def test_add_html(self):
        from scomp_link.mcp_server import _reports, report_add_text

        report_add_text(self.rid, "<div class='custom'>Custom HTML</div>", style="html")
        assert "class='custom'" in _reports[self.rid].html_report

    def test_invalid_report_id(self):
        from scomp_link.mcp_server import report_add_text

        result = json.loads(report_add_text("nonexistent", "text"))
        assert "error" in result

    def teardown_method(self):
        from scomp_link.mcp_server import _reports

        _reports.clear()


class TestReportAddTable:
    """Tests for report_add_table MCP tool."""

    def setup_method(self):
        from scomp_link.mcp_server import _reports, report_add_section, report_create

        _reports.clear()
        result = json.loads(report_create("Test"))
        self.rid = result["report_id"]
        report_add_section(self.rid, "Section")

    def test_add_table(self):
        from scomp_link.mcp_server import report_add_table

        data = json.dumps([{"name": "Alice", "score": 95}, {"name": "Bob", "score": 87}])
        result = json.loads(report_add_table(self.rid, data, "Scores"))
        assert result["status"] == "table_added"
        assert result["rows"] == 2
        assert result["columns"] == 2

    def test_add_table_single_row(self):
        from scomp_link.mcp_server import report_add_table

        data = json.dumps([{"metric": "R2", "value": 0.95}])
        result = json.loads(report_add_table(self.rid, data))
        assert result["rows"] == 1

    def test_invalid_report_id(self):
        from scomp_link.mcp_server import report_add_table

        result = json.loads(report_add_table("nonexistent", "[]"))
        assert "error" in result

    def teardown_method(self):
        from scomp_link.mcp_server import _reports

        _reports.clear()


class TestReportAddChart:
    """Tests for report_add_chart MCP tool."""

    def setup_method(self):
        from scomp_link.mcp_server import _reports, report_add_section, report_create

        _reports.clear()
        result = json.loads(report_create("Test"))
        self.rid = result["report_id"]
        report_add_section(self.rid, "Charts")

    def test_plotly_histogram(self):
        from scomp_link.mcp_server import report_add_chart

        data = json.dumps({"values": [1, 2, 3, 4, 5, 3, 2, 4, 5, 6], "name": "Test"})
        result = json.loads(report_add_chart(self.rid, "plotly", "histogram", data, "Hist"))
        assert result["status"] == "chart_added"
        assert result["engine"] == "plotly"

    def test_plotly_barchart(self):
        from scomp_link.mcp_server import report_add_chart

        data = json.dumps({"categories": ["A", "B", "C"], "values": [[10, 20, 30]], "y_axis_titles": ["Revenue"]})
        result = json.loads(report_add_chart(self.rid, "plotly", "barchart", data, "Bar"))
        assert result["status"] == "chart_added"

    def test_plotly_linechart(self):
        from scomp_link.mcp_server import report_add_chart

        data = json.dumps(
            {"dates": ["2024-01-01", "2024-02-01", "2024-03-01"], "lines": [[100, 120, 140]], "y_labels": ["Sales"]}
        )
        result = json.loads(report_add_chart(self.rid, "plotly", "linechart", data, "Line"))
        assert result["status"] == "chart_added"

    def test_plotly_area_chart(self):
        from scomp_link.mcp_server import report_add_chart

        data = json.dumps(
            {
                "dates": ["2024-01-01", "2024-02-01", "2024-03-01"],
                "lines": [[10, 20, 15], [5, 15, 10]],
                "y_labels": ["A", "B"],
            }
        )
        result = json.loads(report_add_chart(self.rid, "plotly", "area_chart", data, "Area"))
        assert result["status"] == "chart_added"

    def test_rawgraphs_treemap(self):
        from scomp_link.mcp_server import report_add_chart

        data = json.dumps(
            {
                "data": {
                    "name": "root",
                    "children": [
                        {"name": "A", "value": 30},
                        {"name": "B", "value": 20},
                    ],
                }
            }
        )
        result = json.loads(report_add_chart(self.rid, "rawgraphs", "treemap", data, "Treemap"))
        assert result["status"] == "chart_added"
        assert result["engine"] == "rawgraphs"

    def test_rawgraphs_piechart(self):
        from scomp_link.mcp_server import report_add_chart

        data = json.dumps({"labels": ["A", "B", "C"], "values": [40, 35, 25]})
        result = json.loads(report_add_chart(self.rid, "rawgraphs", "piechart", data, "Pie"))
        assert result["status"] == "chart_added"

    def test_rawgraphs_barchart(self):
        from scomp_link.mcp_server import report_add_chart

        data = json.dumps({"categories": ["Q1", "Q2", "Q3"], "values": [100, 150, 120]})
        result = json.loads(report_add_chart(self.rid, "rawgraphs", "barchart", data, "Bar"))
        assert result["status"] == "chart_added"

    def test_highcharts_streamgraphs(self):
        from scomp_link.mcp_server import report_add_chart

        data = json.dumps(
            {
                "dates": ["Jan", "Feb", "Mar"],
                "series_dict": {"A": [10, 20, 15], "B": [5, 15, 10]},
                "area": True,
            }
        )
        result = json.loads(report_add_chart(self.rid, "highcharts", "streamgraphs", data, "Stream"))
        assert result["status"] == "chart_added"
        assert result["engine"] == "highcharts"

    def test_invalid_engine(self):
        from scomp_link.mcp_server import report_add_chart

        result = json.loads(report_add_chart(self.rid, "invalid_engine", "bar", "{}", "X"))
        assert "error" in result
        assert "Unknown engine" in result["error"]

    def test_invalid_plotly_chart_type(self):
        from scomp_link.mcp_server import report_add_chart

        result = json.loads(report_add_chart(self.rid, "plotly", "invalid_type", "{}", "X"))
        assert "error" in result
        assert "Unknown plotly chart_type" in result["error"]

    def test_invalid_rawgraphs_chart_type(self):
        from scomp_link.mcp_server import report_add_chart

        result = json.loads(report_add_chart(self.rid, "rawgraphs", "nonexistent", "{}", "X"))
        assert "error" in result

    def test_invalid_highcharts_chart_type(self):
        from scomp_link.mcp_server import report_add_chart

        result = json.loads(report_add_chart(self.rid, "highcharts", "invalid", "{}", "X"))
        assert "error" in result

    def test_invalid_report_id(self):
        from scomp_link.mcp_server import report_add_chart

        result = json.loads(report_add_chart("nonexistent", "plotly", "histogram", "{}", "X"))
        assert "error" in result

    def test_chart_generation_error(self):
        from scomp_link.mcp_server import report_add_chart

        # Invalid data format should trigger error handling
        result = json.loads(report_add_chart(self.rid, "plotly", "histogram", "{}", "X"))
        assert "error" in result
        assert "Chart generation failed" in result["error"]

    def test_linechart_with_custom_format_date(self):
        from scomp_link.mcp_server import report_add_chart

        data = json.dumps(
            {
                "dates": ["2024-01", "2024-02", "2024-03"],
                "lines": [[100, 120, 140]],
                "y_labels": ["Sales"],
                "format_date": "%Y-%m",
            }
        )
        result = json.loads(report_add_chart(self.rid, "plotly", "linechart", data, "Monthly"))
        assert result["status"] == "chart_added"

    def teardown_method(self):
        from scomp_link.mcp_server import _reports

        _reports.clear()


class TestReportSave:
    """Tests for report_save MCP tool."""

    def setup_method(self):
        from scomp_link.mcp_server import _reports, report_add_section, report_add_text, report_create

        _reports.clear()
        result = json.loads(report_create("Test Report"))
        self.rid = result["report_id"]
        report_add_section(self.rid, "Section")
        report_add_text(self.rid, "Content here")

    def test_save_html(self):
        from scomp_link.mcp_server import _reports, report_save

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            result = json.loads(report_save(self.rid, path))
            assert result["status"] == "saved"
            assert result["path"] == path
            assert result["size_kb"] > 0
            assert os.path.exists(path)
            assert self.rid not in _reports  # Session cleaned up
        finally:
            os.unlink(path)

    def test_save_closes_open_section(self):
        from scomp_link.mcp_server import _reports, report_save

        # Section is open from setup
        assert _reports[self.rid].section_just_open is True
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            report_save(self.rid, path)
            # Verify HTML is valid (section was closed)
            with open(path) as f:
                html = f.read()
            assert "</html>" in html
        finally:
            os.unlink(path)

    def test_save_invalid_report_id(self):
        from scomp_link.mcp_server import report_save

        result = json.loads(report_save("nonexistent", "/tmp/x.html"))
        assert "error" in result

    def teardown_method(self):
        from scomp_link.mcp_server import _reports

        _reports.clear()


class TestReportFullWorkflow:
    """End-to-end test for the complete report builder workflow."""

    def test_full_workflow_with_all_engines(self):
        from scomp_link.mcp_server import (
            _reports,
            report_add_chart,
            report_add_section,
            report_add_table,
            report_add_text,
            report_create,
            report_save,
        )

        _reports.clear()

        # Create with custom branding
        result = json.loads(
            report_create(
                "Pirelli Q4 Report",
                main_color="#CC0000",
                footer_html="<footer><strong>Pirelli S.p.A.</strong></footer>",
            )
        )
        rid = result["report_id"]

        # Add structured content
        report_add_section(rid, "Executive Summary")
        report_add_text(rid, "Q4 2026 analysis.", style="paragraph")
        data = json.dumps([{"product": "P Zero", "revenue": 120}, {"product": "Cinturato", "revenue": 95}])
        report_add_table(rid, data, "Revenue")

        # Plotly chart
        report_add_section(rid, "Distributions")
        chart_data = json.dumps({"values": list(range(50)), "name": "Durability"})
        report_add_chart(rid, "plotly", "histogram", chart_data, "Distribution")

        # RAWGraphs chart
        report_add_section(rid, "Hierarchy")
        tree_data = json.dumps({"data": {"name": "root", "children": [{"name": "A", "value": 60}]}})
        report_add_chart(rid, "rawgraphs", "treemap", tree_data, "Market")

        # Save
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name
        try:
            result = json.loads(report_save(rid, path))
            assert result["status"] == "saved"
            assert result["size_kb"] > 10  # Non-trivial report

            with open(path) as f:
                html = f.read()
            assert "Pirelli S.p.A." in html
            assert "#CC0000" in html
            assert "About scomp-link" not in html
            assert "Executive Summary" in html
            assert rid not in _reports
        finally:
            os.unlink(path)


class TestGenerateReportFooter:
    """Test that generate_report accepts footer_html parameter."""

    def test_generate_report_with_custom_footer(self):
        import numpy as np
        import pandas as pd

        from scomp_link.mcp_server import generate_report

        # Create a small test CSV
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as f:
            df = pd.DataFrame({"a": np.random.rand(20), "b": np.random.rand(20)})
            df.to_csv(f.name, index=False)
            csv_path = f.name

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            html_path = f.name

        try:
            result = json.loads(
                generate_report(
                    csv_path, output=html_path, title="Test", footer_html="<footer>Custom Corp Footer</footer>"
                )
            )
            assert result["report_path"] == html_path
            with open(html_path) as f:
                html = f.read()
            assert "Custom Corp Footer" in html
        finally:
            os.unlink(csv_path)
            os.unlink(html_path)


class TestReportAddChartHighcharts:
    """Additional Highcharts tests for coverage."""

    def setup_method(self):
        from scomp_link.mcp_server import _reports, report_add_section, report_create

        _reports.clear()
        result = json.loads(report_create("Test"))
        self.rid = result["report_id"]
        report_add_section(self.rid, "HC")

    def test_calendar_heatmap(self):
        from scomp_link.mcp_server import report_add_chart

        data = json.dumps(
            {
                "series_dict": {
                    "2024-01-01": 0.8,
                    "2024-01-02": 0.6,
                    "2024-01-03": 0.9,
                    "2024-01-04": 0.5,
                    "2024-01-05": 0.7,
                    "2024-01-06": 0.3,
                    "2024-01-07": 0.95,
                },
                "min": 0,
                "max": 1,
            }
        )
        result = json.loads(report_add_chart(self.rid, "highcharts", "calendar_heatmap", data, "Heatmap"))
        assert result["status"] == "chart_added"

    def teardown_method(self):
        from scomp_link.mcp_server import _reports

        _reports.clear()


class TestReportAddChartRawgraphsExtra:
    """Additional RAWGraphs chart tests for coverage."""

    def setup_method(self):
        from scomp_link.mcp_server import _reports, report_add_section, report_create

        _reports.clear()
        result = json.loads(report_create("Test"))
        self.rid = result["report_id"]
        report_add_section(self.rid, "RG")

    def test_sunburst(self):
        from scomp_link.mcp_server import report_add_chart

        data = json.dumps(
            {
                "data": {
                    "name": "root",
                    "children": [
                        {"name": "A", "children": [{"name": "A1", "value": 10}, {"name": "A2", "value": 15}]},
                        {"name": "B", "value": 20},
                    ],
                }
            }
        )
        result = json.loads(report_add_chart(self.rid, "rawgraphs", "sunburst", data, "Sunburst"))
        assert result["status"] == "chart_added"

    def test_boxplot(self):
        import numpy as np

        from scomp_link.mcp_server import report_add_chart

        np.random.seed(42)
        data = json.dumps(
            {
                "data": [np.random.normal(50, 10, 30).tolist(), np.random.normal(60, 15, 30).tolist()],
                "labels": ["Group A", "Group B"],
            }
        )
        result = json.loads(report_add_chart(self.rid, "rawgraphs", "boxplot", data, "Boxplot"))
        assert result["status"] == "chart_added"

    def test_matrixplot(self):
        from scomp_link.mcp_server import report_add_chart

        data = json.dumps(
            {
                "matrix": [[1.0, 0.8, 0.3], [0.8, 1.0, 0.5], [0.3, 0.5, 1.0]],
                "row_labels": ["A", "B", "C"],
                "col_labels": ["A", "B", "C"],
            }
        )
        result = json.loads(report_add_chart(self.rid, "rawgraphs", "matrixplot", data, "Correlation"))
        assert result["status"] == "chart_added"

    def test_sankey(self):
        from scomp_link.mcp_server import report_add_chart

        data = json.dumps(
            {
                "nodes": [{"name": "A", "x": 0}, {"name": "B", "x": 1}, {"name": "C", "x": 2}],
                "links": [{"source": 0, "target": 1, "value": 30}, {"source": 1, "target": 2, "value": 20}],
            }
        )
        result = json.loads(report_add_chart(self.rid, "rawgraphs", "sankey_diagram", data, "Flow"))
        assert result["status"] == "chart_added"

    def test_radarchart(self):
        from scomp_link.mcp_server import report_add_chart

        data = json.dumps(
            {
                "categories": ["Speed", "Power", "Range", "Comfort", "Safety"],
                "series": {"Model X": [8, 6, 9, 7, 8], "Model Y": [7, 8, 6, 9, 7]},
            }
        )
        result = json.loads(report_add_chart(self.rid, "rawgraphs", "radarchart", data, "Radar"))
        assert result["status"] == "chart_added"

    def test_linechart_rawgraphs(self):
        from scomp_link.mcp_server import report_add_chart

        data = json.dumps(
            {
                "series": {"Revenue": [100, 120, 140, 160], "Cost": [80, 85, 90, 95]},
                "x_values": ["Q1", "Q2", "Q3", "Q4"],
            }
        )
        result = json.loads(report_add_chart(self.rid, "rawgraphs", "linechart", data, "Trend"))
        assert result["status"] == "chart_added"

    def teardown_method(self):
        from scomp_link.mcp_server import _reports

        _reports.clear()


class TestBuildFooterHelper:
    """Test the _build_footer module-level helper."""

    def test_build_footer_default(self):
        from scomp_link.utils.report_html import _DEFAULT_FOOTER_CONTENT, _FOOTER_JS_BLOCK, _build_footer

        result = _build_footer()
        assert _FOOTER_JS_BLOCK in result
        assert _DEFAULT_FOOTER_CONTENT in result

    def test_build_footer_custom(self):
        from scomp_link.utils.report_html import _DEFAULT_FOOTER_CONTENT, _FOOTER_JS_BLOCK, _build_footer

        custom = "<footer>Custom</footer>"
        result = _build_footer(custom)
        assert _FOOTER_JS_BLOCK in result
        assert custom in result
        assert _DEFAULT_FOOTER_CONTENT not in result


# ═══════════════════════════════════════════════════════════════════
# __init_subclass__ Tests
# ═══════════════════════════════════════════════════════════════════


class TestInitSubclass:
    """Tests for ScompLinkHTMLReport.__init_subclass__ (class-level configuration)."""

    def test_subclass_basic(self):
        from scomp_link.utils.report_html import ScompLinkHTMLReport

        class MyReport(ScompLinkHTMLReport, title="Custom Report"):
            pass

        assert MyReport.html_title == "<title>Custom Report</title>"
        assert "Custom Report" in MyReport.header
        assert MyReport.main_color == "#6E37FA"  # default
        assert MyReport.lan == "en"

    def test_subclass_custom_colors(self):
        from scomp_link.utils.report_html import ScompLinkHTMLReport

        class PirelliReport(
            ScompLinkHTMLReport,
            title="Pirelli Analysis",
            main_color="#CC0000",
            light_color="#FF3333",
            dark_color="#990000",
        ):
            pass

        assert PirelliReport.main_color == "#CC0000"
        assert PirelliReport.light_color == "#FF3333"
        assert PirelliReport.dark_color == "#990000"
        assert "#CC0000" in PirelliReport.html_layout

    def test_subclass_custom_author_description(self):
        from scomp_link.utils.report_html import ScompLinkHTMLReport

        class CorpReport(
            ScompLinkHTMLReport, title="Corp Report", description="Internal Analytics", author="Pirelli Digital"
        ):
            pass

        assert "Internal Analytics" in CorpReport.html_meta_info
        assert "Pirelli Digital" in CorpReport.html_meta_info

    def test_subclass_custom_font(self):
        from scomp_link.utils.report_html import ScompLinkHTMLReport

        class ArialReport(ScompLinkHTMLReport, title="Arial Report", font_family="Arial"):
            pass

        assert "Arial" in ArialReport.html_layout

    def test_subclass_custom_language(self):
        from scomp_link.utils.report_html import ScompLinkHTMLReport

        class ItalianReport(ScompLinkHTMLReport, title="Rapporto", language="it"):
            pass

        assert ItalianReport.lan == "it"

    def test_subclass_custom_footer(self):
        from scomp_link.utils.report_html import _DEFAULT_FOOTER_CONTENT, ScompLinkHTMLReport

        custom_footer = "<footer><strong>Pirelli S.p.A.</strong><br>Confidential.</footer>"

        class PirelliReport(ScompLinkHTMLReport, title="Pirelli", footer_html=custom_footer):
            pass

        assert custom_footer in PirelliReport.footer
        assert _DEFAULT_FOOTER_CONTENT not in PirelliReport.footer
        # JS still present
        assert "download_table_as_csv" in PirelliReport.footer

    def test_subclass_default_footer(self):
        from scomp_link.utils.report_html import _DEFAULT_FOOTER_CONTENT, ScompLinkHTMLReport

        class DefaultReport(ScompLinkHTMLReport, title="Default"):
            pass

        assert _DEFAULT_FOOTER_CONTENT in DefaultReport.footer

    def test_subclass_custom_logo_and_background(self):
        from scomp_link.utils.report_html import ScompLinkHTMLReport

        class BrandedReport(
            ScompLinkHTMLReport,
            title="Branded",
            url_img_logo="https://example.com/logo.png",
            url_background_header="https://example.com/bg.jpg",
        ):
            pass

        assert "https://example.com/logo.png" in BrandedReport.html_layout
        assert "https://example.com/bg.jpg" in BrandedReport.html_layout

    def test_subclass_instance_has_class_attrs(self):
        """Verify that instances of the subclass inherit class attributes."""
        from scomp_link.utils.report_html import ScompLinkHTMLReport

        class TestReport(ScompLinkHTMLReport, title="Subclass Test"):
            pass

        # The subclass attributes are class-level, but still accessible
        assert TestReport.html_title == "<title>Subclass Test</title>"
        assert hasattr(TestReport, "footer")
        assert hasattr(TestReport, "html_layout")

    def test_subclass_html_report_and_section_state(self):
        from scomp_link.utils.report_html import ScompLinkHTMLReport

        class StateReport(ScompLinkHTMLReport, title="State Test"):
            pass

        assert StateReport.html_report == ""
        assert StateReport.section_just_open is False


# ═══════════════════════════════════════════════════════════════════
# PDF Converter Tests (mocked where WeasyPrint unavailable)
# ═══════════════════════════════════════════════════════════════════


class TestPdfConverter:
    """Tests for scomp_link.utils.pdf_converter."""

    def test_wrap_html_default_css(self):
        from scomp_link.utils.pdf_converter import _wrap_html

        result = _wrap_html("<p>Hello</p>")
        assert "<!DOCTYPE html>" in result
        assert "<p>Hello</p>" in result
        assert "font-family" in result  # default CSS applied

    def test_wrap_html_custom_css(self):
        from scomp_link.utils.pdf_converter import _wrap_html

        custom_css = "body { color: red; }"
        result = _wrap_html("<p>Test</p>", css=custom_css)
        assert custom_css in result
        assert "<p>Test</p>" in result

    def test_wrap_html_contains_color_constants(self):
        from scomp_link.utils.colors import MAIN, MAIN_DARK
        from scomp_link.utils.pdf_converter import _wrap_html

        result = _wrap_html("<p>Test</p>")
        assert MAIN in result
        assert MAIN_DARK in result

    def test_markdown_to_pdf_no_weasyprint(self):
        """Test that ImportError is raised if weasyprint is not available."""
        from scomp_link.utils import pdf_converter

        original = pdf_converter.HAS_WEASYPRINT
        pdf_converter.HAS_WEASYPRINT = False
        try:
            with pytest.raises(ImportError, match="weasyprint"):
                pdf_converter.markdown_to_pdf("fake.md")
        finally:
            pdf_converter.HAS_WEASYPRINT = original

    def test_markdown_to_pdf_no_markdown(self):
        """Test that ImportError is raised if markdown is not available."""
        from scomp_link.utils import pdf_converter

        original_md = pdf_converter.HAS_MARKDOWN
        original_wp = pdf_converter.HAS_WEASYPRINT
        pdf_converter.HAS_MARKDOWN = False
        pdf_converter.HAS_WEASYPRINT = True
        try:
            with pytest.raises(ImportError, match="markdown"):
                pdf_converter.markdown_to_pdf("fake.md")
        finally:
            pdf_converter.HAS_MARKDOWN = original_md
            pdf_converter.HAS_WEASYPRINT = original_wp

    def test_html_to_pdf_no_weasyprint(self):
        """Test that ImportError is raised if weasyprint is not available."""
        from scomp_link.utils import pdf_converter

        original = pdf_converter.HAS_WEASYPRINT
        pdf_converter.HAS_WEASYPRINT = False
        try:
            with pytest.raises(ImportError, match="weasyprint"):
                pdf_converter.html_to_pdf("fake.html")
        finally:
            pdf_converter.HAS_WEASYPRINT = original

    def test_markdown_to_pdf_default_output_path(self):
        """Test default output path generation (input.md → input.pdf)."""
        import unittest.mock as mock

        from scomp_link.utils import pdf_converter

        original_md = pdf_converter.HAS_MARKDOWN
        original_wp = pdf_converter.HAS_WEASYPRINT
        pdf_converter.HAS_MARKDOWN = True
        pdf_converter.HAS_WEASYPRINT = True

        class MockHTML:
            def __init__(self, **kwargs):
                pass

            def write_pdf(self, path):
                with open(path, "w") as f:
                    f.write("fake pdf")

        # Inject mock HTML into the module namespace
        pdf_converter.HTML = MockHTML

        with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w") as f:
            f.write("# Test\n\nHello world")
            md_path = f.name

        try:
            with mock.patch("markdown.markdown", return_value="<h1>Test</h1><p>Hello world</p>"):
                result = pdf_converter.markdown_to_pdf(md_path)
                assert result == md_path.rsplit(".", 1)[0] + ".pdf"
                assert os.path.exists(result)
                os.unlink(result)
        finally:
            pdf_converter.HAS_MARKDOWN = original_md
            pdf_converter.HAS_WEASYPRINT = original_wp
            if (
                hasattr(pdf_converter, "HTML")
                and isinstance(pdf_converter.HTML, type)
                and pdf_converter.HTML is MockHTML
            ):
                delattr(pdf_converter, "HTML")
            os.unlink(md_path)

    def test_html_to_pdf_with_css_injection(self):
        """Test CSS injection into HTML before PDF conversion."""
        from scomp_link.utils import pdf_converter

        original_wp = pdf_converter.HAS_WEASYPRINT
        pdf_converter.HAS_WEASYPRINT = True

        class MockHTML:
            def __init__(self, **kwargs):
                self.content = kwargs.get("string", "") or ""

            def write_pdf(self, path):
                with open(path, "w") as f:
                    f.write(self.content)

        pdf_converter.HTML = MockHTML

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
            f.write("<html><head></head><body><p>Test</p></body></html>")
            html_path = f.name

        try:
            out_path = html_path.replace(".html", ".pdf")
            result = pdf_converter.html_to_pdf(html_path, css="body { color: blue; }")
            assert result == out_path
            with open(out_path) as f:
                content = f.read()
            assert "color: blue" in content
            os.unlink(out_path)
        finally:
            pdf_converter.HAS_WEASYPRINT = original_wp
            delattr(pdf_converter, "HTML")
            os.unlink(html_path)

    def test_html_to_pdf_without_css(self):
        """Test HTML to PDF without custom CSS."""
        from scomp_link.utils import pdf_converter

        original_wp = pdf_converter.HAS_WEASYPRINT
        pdf_converter.HAS_WEASYPRINT = True

        class MockHTML:
            def __init__(self, **kwargs):
                pass

            def write_pdf(self, path):
                with open(path, "w") as f:
                    f.write("pdf content")

        pdf_converter.HTML = MockHTML

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
            f.write("<html><body>Hello</body></html>")
            html_path = f.name

        try:
            out_path = html_path.replace(".html", ".pdf")
            result = pdf_converter.html_to_pdf(html_path)
            assert result == out_path
            os.unlink(out_path)
        finally:
            pdf_converter.HAS_WEASYPRINT = original_wp
            delattr(pdf_converter, "HTML")
            os.unlink(html_path)

    def test_html_to_pdf_css_no_head_tag(self):
        """Test CSS injection when HTML has no </head> tag."""
        from scomp_link.utils import pdf_converter

        original_wp = pdf_converter.HAS_WEASYPRINT
        pdf_converter.HAS_WEASYPRINT = True

        class MockHTML:
            def __init__(self, **kwargs):
                self.content = kwargs.get("string", "")

            def write_pdf(self, path):
                with open(path, "w") as f:
                    f.write(self.content)

        pdf_converter.HTML = MockHTML

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
            f.write("<body><p>No head</p></body>")
            html_path = f.name

        try:
            out_path = html_path.replace(".html", ".pdf")
            pdf_converter.html_to_pdf(html_path, css="p { color: green; }")
            with open(out_path) as f:
                content = f.read()
            assert "color: green" in content
            os.unlink(out_path)
        finally:
            pdf_converter.HAS_WEASYPRINT = original_wp
            delattr(pdf_converter, "HTML")
            os.unlink(html_path)


# ═══════════════════════════════════════════════════════════════════
# MCP Prompts Tests
# ═══════════════════════════════════════════════════════════════════


class TestMCPPrompts:
    """Tests for MCP prompt templates."""

    def test_ml_workflow_prompt(self):
        from scomp_link.mcp_server import ml_workflow

        result = ml_workflow("data.csv", "price", "regression")
        assert "data.csv" in result
        assert "price" in result
        assert "regression" in result
        assert "describe_data" in result
        assert "train_model" in result

    def test_ml_workflow_classification(self):
        from scomp_link.mcp_server import ml_workflow

        result = ml_workflow("iris.csv", "species", "classification")
        assert "iris.csv" in result
        assert "species" in result
        assert "classification" in result

    def test_debug_model_prompt(self):
        from scomp_link.mcp_server import debug_model

        result = debug_model("model.scomp", "low accuracy")
        assert "model.scomp" in result
        assert "low accuracy" in result
        assert "detect_drift" in result
        assert "feature engineering" in result

    def test_debug_model_default_issue(self):
        from scomp_link.mcp_server import debug_model

        result = debug_model("model.scomp")
        assert "low accuracy" in result  # default issue

    def test_monitor_production_prompt(self):
        from scomp_link.mcp_server import monitor_production

        result = monitor_production("train.csv", "prod.csv", "model.scomp")
        assert "train.csv" in result
        assert "prod.csv" in result
        assert "model.scomp" in result
        assert "detect_drift" in result
        assert "PSI" in result

    def test_monitor_production_no_artifact(self):
        from scomp_link.mcp_server import monitor_production

        result = monitor_production("train.csv", "prod.csv")
        assert "not provided" in result

    def test_create_dashboard_prompt(self):
        from scomp_link.mcp_server import create_dashboard

        result = create_dashboard("sales.csv", "Sales Dashboard")
        assert "sales.csv" in result
        assert "Sales Dashboard" in result
        assert "describe_data" in result
        assert "Plotly" in result
        assert "RAWGraphs" in result
        assert "Highcharts" in result

    def test_create_dashboard_default_title(self):
        from scomp_link.mcp_server import create_dashboard

        result = create_dashboard("data.csv")
        assert "Dashboard" in result

    def test_build_custom_report_prompt(self):
        from scomp_link.mcp_server import build_custom_report

        result = build_custom_report("analysis.csv", "Q4 Report")
        assert "analysis.csv" in result
        assert "Q4 Report" in result
        assert "report_create" in result
        assert "report_add_section" in result
        assert "report_add_chart" in result
        assert "report_save" in result
        assert "config" in result.lower()

    def test_build_custom_report_default_title(self):
        from scomp_link.mcp_server import build_custom_report

        result = build_custom_report("data.csv")
        assert "Report" in result
