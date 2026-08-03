# -*- coding: utf-8 -*-
"""Tests for the >> DSL in scomp_link/pipeline_dsl.py."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from scomp_link import (
    Chain,
    CleanStep,
    GraphStep,
    LogStep,
    ModelStep,
    RawGraphStep,
    SaveStep,
    ScompLinkPipeline,
    SectionStep,
    SelectStep,
    TableStep,
    TextStep,
    TitleStep,
    TrainStep,
)
from scomp_link.pipeline_dsl import MLStep, ReportStep, Step
from scomp_link.utils.report_html import ScompLinkHTMLReport

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_df():
    np.random.seed(42)
    return pd.DataFrame(
        {
            "x1": np.random.randn(200),
            "x2": np.random.randn(200),
            "y": np.random.randn(200),
        }
    )


@pytest.fixture
def metrics_df():
    return pd.DataFrame({"metric": ["RMSE", "MAE"], "value": [0.81, 0.65]})


@pytest.fixture
def report():
    return ScompLinkHTMLReport("Test Report")


# ---------------------------------------------------------------------------
# Chain construction
# ---------------------------------------------------------------------------


def test_rshift_returns_chain(small_df):
    chain = CleanStep(small_df) >> SelectStep("y")
    assert isinstance(chain, Chain)


def test_chain_repr(small_df):
    chain = CleanStep(small_df) >> SelectStep("y") >> TrainStep("regression")
    assert "CleanStep" in repr(chain)
    assert "SelectStep" in repr(chain)
    assert "TrainStep" in repr(chain)
    assert ">>" in repr(chain)


def test_chain_step_count(small_df):
    chain = CleanStep(small_df) >> SelectStep("y") >> ModelStep("numerical_prediction") >> TrainStep("regression")
    assert len(chain._steps) == 4


def test_chain_chain_rshift(small_df):
    """Chaining two Chain objects with >> should flatten into one Chain."""
    a = CleanStep(small_df) >> SelectStep("y")
    b = ModelStep("numerical_prediction") >> TrainStep("regression")
    combined = a >> b
    assert len(combined._steps) == 4


def test_mix_ml_report_raises(small_df):
    """Mixing MLStep and ReportStep in the same chain must raise TypeError immediately."""
    with pytest.raises(TypeError, match="Cannot mix"):
        _ = CleanStep(small_df) >> TitleStep("oops")


def test_mix_report_ml_raises(small_df):
    with pytest.raises(TypeError, match="Cannot mix"):
        _ = TitleStep("oops") >> CleanStep(small_df)


def test_empty_chain_raises():
    with pytest.raises(ValueError):
        Chain([])


# ---------------------------------------------------------------------------
# ML chain execution
# ---------------------------------------------------------------------------


def test_ml_chain_basic(small_df):
    results = (
        CleanStep(small_df) >> SelectStep("y") >> ModelStep("numerical_prediction") >> TrainStep("regression")
    ).run()
    assert results["status"] == "success"
    assert "metrics" in results


def test_ml_chain_run_with_pipeline_arg(small_df):
    """Chain.run(pipeline) should use the provided ScompLinkPipeline."""
    pipeline = ScompLinkPipeline("arg pipeline")
    pipeline.import_and_clean_data(small_df)
    results = (SelectStep("y") >> ModelStep("numerical_prediction") >> TrainStep("regression")).run(pipeline)
    assert results["status"] == "success"


def test_ml_chain_existing_pipeline(small_df):
    """pipeline >> step should carry the existing pipeline through the chain."""
    pipeline = ScompLinkPipeline("existing")
    pipeline.import_and_clean_data(small_df)
    results = (pipeline >> SelectStep("y") >> ModelStep("numerical_prediction") >> TrainStep("regression")).run()
    assert results["status"] == "success"


def test_clean_step_no_df_no_target_raises():
    """CleanStep with no df and no target to .run() must raise."""
    with pytest.raises((ValueError, TypeError)):
        (CleanStep() >> SelectStep("y") >> TrainStep("regression")).run()


# ---------------------------------------------------------------------------
# Report chain execution
# ---------------------------------------------------------------------------


def test_report_chain_basic(metrics_df, report, tmp_path):
    out = str(tmp_path / "out.html")
    result = (
        SectionStep("Results")
        >> TitleStep("KPIs")
        >> TextStep("Some explanation.")
        >> TableStep(metrics_df, "Metrics")
        >> SaveStep(out)
    ).run(report)
    assert os.path.exists(out)
    assert isinstance(result, ScompLinkHTMLReport)


def test_report_chain_no_target(tmp_path):
    """chain.run() with no arg should auto-create a default ScompLinkHTMLReport."""
    out = str(tmp_path / "auto.html")
    (SectionStep("Auto") >> TextStep("hello") >> SaveStep(out)).run()
    assert os.path.exists(out)


def test_section_step_auto_close(report, tmp_path):
    """SectionStep should auto-close any previously open section."""
    out = str(tmp_path / "sections.html")
    (
        SectionStep("First")
        >> TextStep("in first")
        >> SectionStep("Second")  # should auto-close First
        >> TextStep("in second")
        >> SaveStep(out)
    ).run(report)
    assert os.path.exists(out)
    # After SaveStep, no section should be open
    assert not report.section_just_open


def test_save_step_closes_section(report, tmp_path):
    """SaveStep should close any open section before saving."""
    out = str(tmp_path / "close.html")
    (SectionStep("Open") >> TextStep("content") >> SaveStep(out)).run(report)
    assert not report.section_just_open
    assert os.path.exists(out)


# ---------------------------------------------------------------------------
# LogStep
# ---------------------------------------------------------------------------


def test_log_step_ml(small_df, caplog):
    """LogStep should pass through the pipeline unchanged."""
    import logging

    with caplog.at_level(logging.INFO):
        results = (
            CleanStep(small_df)
            >> LogStep("after clean")
            >> SelectStep("y")
            >> LogStep("after select")
            >> ModelStep("numerical_prediction")
            >> TrainStep("regression")
        ).run()
    assert results["status"] == "success"


def test_log_step_report(metrics_df, report, tmp_path):
    """LogStep should pass through the report unchanged."""
    out = str(tmp_path / "log.html")
    (SectionStep("Results") >> LogStep("check report state") >> TableStep(metrics_df, "Metrics") >> SaveStep(out)).run(
        report
    )
    assert os.path.exists(out)


def test_log_step_does_not_count_as_ml_or_report(small_df):
    """LogStep mixed with either ML or Report steps should not trigger the homogeneity error."""
    # Should not raise
    chain = CleanStep(small_df) >> LogStep() >> SelectStep("y") >> TrainStep("regression")
    assert isinstance(chain, Chain)


def test_log_step_as_first_step_ml(small_df):
    """Chain starting with LogStep should still execute correctly as ML chain."""
    results = (
        LogStep("before clean")
        >> CleanStep(small_df)
        >> LogStep("after clean")
        >> SelectStep("y")
        >> ModelStep("numerical_prediction")
        >> TrainStep("regression")
    ).run()
    assert results["status"] == "success"


def test_log_step_as_first_step_report(metrics_df, report, tmp_path):
    """Chain starting with LogStep should still execute correctly as Report chain."""
    out = str(tmp_path / "log_first.html")
    (LogStep("before section") >> SectionStep("Results") >> TableStep(metrics_df, "Metrics") >> SaveStep(out)).run(
        report
    )
    assert os.path.exists(out)


def test_only_log_steps_raises():
    """Chain with only LogStep instances should raise TypeError on .run()."""
    with pytest.raises(TypeError, match="only LogStep"):
        (LogStep() >> LogStep()).run()
