# -*- coding: utf-8 -*-
"""
██████╗ ███████╗██╗         ██████╗ ██████╗ ███████╗██████╗  █████╗ ████████╗ ██████╗ ██████╗
██╔══██╗██╔════╝██║         ██╔═══██╗██╔══██╗██╔════╝██╔══██╗██╔══██╗╚══██╔══╝██╔═══██╗██╔══██╗
██║  ██║███████╗██║         ██║   ██║██████╔╝█████╗  ██████╔╝███████║   ██║   ██║   ██║██████╔╝
██║  ██║╚════██║██║         ██║   ██║██╔═══╝ ██╔══╝  ██╔══██╗██╔══██║   ██║   ██║   ██║██╔══██╗
██████╔╝███████║███████╗    ╚██████╔╝██║     ███████╗██║  ██║██║  ██║   ██║   ╚██████╔╝██║  ██║
╚═════╝ ╚══════╝╚══════╝     ╚═════╝ ╚═╝     ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝

Airflow-style >> DSL for ScompLinkPipeline and ScompLinkHTMLReport.

Usage — ML pipeline:
    from scomp_link import CleanStep, SelectStep, ModelStep, TrainStep

    pipe = CleanStep(df) >> SelectStep("price") >> ModelStep("numerical_prediction") >> TrainStep("regression")
    results = pipe.run()

Usage — Report pipeline:
    from scomp_link import SectionStep, TitleStep, TableStep, GraphStep, SaveStep
    from scomp_link.utils.report_html import ScompLinkHTMLReport

    report = ScompLinkHTMLReport("My Report")
    (SectionStep("Results") >> TitleStep("Metrics") >> TableStep(df, "KPIs") >> SaveStep("out.html")).run(report)

The type of chain (ML vs Report) is inferred from the first step.
If the first step is an MLStep, .run() creates or accepts a ScompLinkPipeline.
If the first step is a ReportStep, .run() creates or accepts a ScompLinkHTMLReport.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pandas as pd

    from scomp_link.core import ScompLinkPipeline
    from scomp_link.utils.report_html import ScompLinkHTMLReport

from scomp_link.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Base classes
# ---------------------------------------------------------------------------


class Step(ABC):
    """Abstract base step. Subclass as MLStep or ReportStep."""

    def __rshift__(self, other: "Step | Chain") -> "Chain":
        if isinstance(other, Chain):
            return Chain([self, *other._steps])
        return Chain([self, other])

    @abstractmethod
    def execute(self, target: Any) -> Any:
        """Execute this step against target, return (possibly updated) target."""


class Chain:
    """
    An ordered sequence of Steps built with >>. Lazy — nothing runs until .run().

    The chain type (ML or Report) is determined by the first step:
    - First step is MLStep  → .run() manages a ScompLinkPipeline
    - First step is ReportStep → .run() manages a ScompLinkHTMLReport
    """

    def __init__(self, steps: list[Step]):
        if not steps:
            raise ValueError("Chain must contain at least one step.")
        # Validate homogeneity: cannot mix MLStep and ReportStep in the same chain
        has_ml = any(isinstance(s, MLStep) for s in steps if not isinstance(s, LogStep))
        has_report = any(isinstance(s, ReportStep) for s in steps if not isinstance(s, LogStep))
        if has_ml and has_report:
            ml_names = [type(s).__name__ for s in steps if isinstance(s, MLStep)]
            rep_names = [type(s).__name__ for s in steps if isinstance(s, ReportStep)]
            raise TypeError(
                f"Cannot mix MLStep ({ml_names}) and ReportStep ({rep_names}) in the same chain. "
                "Use separate chains for ML and report building."
            )
        self._steps = steps

    def __rshift__(self, other: "Step | Chain") -> "Chain":
        if isinstance(other, Chain):
            return Chain([*self._steps, *other._steps])
        return Chain([*self._steps, other])

    def run(self, target: Any = None) -> Any:
        """
        Execute the chain.

        For ML chains:   pass a pd.DataFrame or ScompLinkPipeline (or nothing — uses CleanStep's df).
        For Report chains: pass a ScompLinkHTMLReport (or nothing — a default report is created).

        Returns the final result: results dict for ML chains, ScompLinkHTMLReport for report chains.
        """
        # Infer chain type from the first non-LogStep step
        typed_step = next((s for s in self._steps if not isinstance(s, LogStep)), None)
        if typed_step is None:
            raise TypeError("Chain contains only LogStep instances — cannot infer execution type.")
        if isinstance(typed_step, MLStep):
            return self._run_ml(target)
        if isinstance(typed_step, ReportStep):
            return self._run_report(target)
        raise TypeError(f"Cannot infer chain type from first typed step {type(typed_step).__name__}")

    def _run_ml(self, target: Any) -> Any:
        import pandas as pd

        from scomp_link.core import ScompLinkPipeline

        # Find the first non-LogStep to determine the pipeline source
        first_typed_idx = next((i for i, s in enumerate(self._steps) if not isinstance(s, LogStep)), 0)
        first_typed = self._steps[first_typed_idx]

        if isinstance(first_typed, _BoundPipelineStep):
            pipeline = first_typed._pipeline
            steps_to_run = self._steps[first_typed_idx + 1 :]
            # Re-prepend any leading LogSteps so they still execute
            steps_to_run = list(self._steps[:first_typed_idx]) + list(steps_to_run)
            # Warn if a CleanStep appears after a bound pipeline (would overwrite its data)
            for s in steps_to_run:
                if isinstance(s, CleanStep):
                    logger.warning(
                        "CleanStep found after an already-initialized pipeline. "
                        "It will overwrite the pipeline's data. "
                        "Remove CleanStep or start from CleanStep(df) instead."
                    )
                    break
        elif target is None:
            # Try to get df from CleanStep
            if isinstance(first_typed, CleanStep) and first_typed.df is not None:
                pipeline = ScompLinkPipeline("DSL Pipeline")
                steps_to_run = self._steps
            else:
                raise ValueError(
                    "ML chain requires a DataFrame or ScompLinkPipeline. "
                    "Pass it to .run(df) or embed it in CleanStep(df)."
                )
        elif isinstance(target, pd.DataFrame):
            pipeline = ScompLinkPipeline("DSL Pipeline")
            steps_to_run = self._steps
        elif isinstance(target, ScompLinkPipeline):
            pipeline = target
            steps_to_run = self._steps
        else:
            raise TypeError(f"ML chain expects DataFrame or ScompLinkPipeline, got {type(target)}")

        result: Any = pipeline
        for step in steps_to_run:
            result = step.execute(result)

        return result

    def _run_report(self, target: Any) -> "ScompLinkHTMLReport":
        from scomp_link.utils.report_html import ScompLinkHTMLReport

        if target is None:
            report = ScompLinkHTMLReport("DSL Report")
        elif isinstance(target, ScompLinkHTMLReport):
            report = target
        else:
            raise TypeError(f"Report chain expects ScompLinkHTMLReport, got {type(target)}")

        for step in self._steps:
            step.execute(report)

        return report

    def __repr__(self) -> str:
        names = " >> ".join(type(s).__name__ for s in self._steps)
        return f"Chain({names})"


# ---------------------------------------------------------------------------
# ML Steps
# ---------------------------------------------------------------------------


class MLStep(Step, ABC):
    """Marker base for steps that operate on ScompLinkPipeline."""


class CleanStep(MLStep):
    """
    Wraps ScompLinkPipeline.import_and_clean_data().

    Parameters
    ----------
    df : pd.DataFrame | None
        The dataframe to load. If not provided here, must be passed to chain.run(df).

    Example
    -------
    CleanStep(df) >> SelectStep("price") >> TrainStep("regression")
    """

    def __init__(self, df: "pd.DataFrame | None" = None):
        self.df = df

    def execute(self, pipeline: "ScompLinkPipeline") -> "ScompLinkPipeline":
        import pandas as pd

        df = self.df
        if df is None:
            raise ValueError("CleanStep requires a DataFrame. Pass it as CleanStep(df).")
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"CleanStep expects a pd.DataFrame, got {type(df)}")
        pipeline.import_and_clean_data(df)
        return pipeline


class SelectStep(MLStep):
    """
    Wraps ScompLinkPipeline.select_variables().

    Parameters
    ----------
    target : str
        Name of the target column.
    features : list[str] | None
        Feature columns. If None, all non-target columns are used.

    Example
    -------
    SelectStep("price", features=["sqm", "rooms", "floor"])
    """

    def __init__(self, target: str, features: "list[str] | None" = None):
        self.target = target
        self.features = features

    def execute(self, pipeline: "ScompLinkPipeline") -> "ScompLinkPipeline":
        pipeline.select_variables(self.target, self.features)
        return pipeline


class ModelStep(MLStep):
    """
    Wraps ScompLinkPipeline.choose_model().

    Parameters
    ----------
    objective : str
        One of: "numerical_prediction", "categorical_known", "categorical_unknown",
        "numerical_study", "multi_numerical_prediction".
    metadata : dict | None
        Extra hints for model selection (e.g. {"only_numerical_exogenous": True}).

    Example
    -------
    ModelStep("numerical_prediction", metadata={"only_numerical_exogenous": True})
    """

    def __init__(self, objective: str, metadata: "dict[str, Any] | None" = None):
        self.objective = objective
        self.metadata = metadata or {}

    def execute(self, pipeline: "ScompLinkPipeline") -> "ScompLinkPipeline":
        pipeline.choose_model(self.objective, self.metadata)
        return pipeline


class TrainStep(MLStep):
    """
    Wraps ScompLinkPipeline.run_pipeline(). This is the terminal ML step — returns the results dict.

    Parameters
    ----------
    task : str
        "regression", "classification", "clustering", "text", "images"
    **kwargs
        Any keyword argument accepted by run_pipeline() (e.g. test_size, use_ensemble, advanced_cv).

    Example
    -------
    TrainStep("regression", test_size=0.25, use_ensemble=True)
    """

    def __init__(self, task: str = "regression", **kwargs: Any):
        self.task = task
        self.kwargs = kwargs

    def execute(self, pipeline: "ScompLinkPipeline") -> Any:
        return pipeline.run_pipeline(task_type=self.task, **self.kwargs)


# ---------------------------------------------------------------------------
# Report Steps
# ---------------------------------------------------------------------------


class ReportStep(Step, ABC):
    """Marker base for steps that operate on ScompLinkHTMLReport."""


class SectionStep(ReportStep):
    """
    Opens a collapsible section. Automatically closes any previously open section.

    Parameters
    ----------
    title : str
        Section heading.

    Example
    -------
    SectionStep("Model Results") >> TableStep(df, "Metrics") >> SectionStep("Charts") >> GraphStep(fig, "ROC")
    """

    def __init__(self, title: str):
        self.title = title

    def execute(self, report: "ScompLinkHTMLReport") -> "ScompLinkHTMLReport":
        if report.section_just_open:
            report.close_section()
        report.open_section(self.title)
        return report


class TitleStep(ReportStep):
    """Adds an <h2> title to the report."""

    def __init__(self, title: str):
        self.title = title

    def execute(self, report: "ScompLinkHTMLReport") -> "ScompLinkHTMLReport":
        report.add_title(self.title)
        return report


class SubtitleStep(ReportStep):
    """Adds an <h3> subtitle to the report."""

    def __init__(self, subtitle: str):
        self.subtitle = subtitle

    def execute(self, report: "ScompLinkHTMLReport") -> "ScompLinkHTMLReport":
        report.add_subtitle(self.subtitle)
        return report


class TextStep(ReportStep):
    """Adds a <p> paragraph to the report."""

    def __init__(self, text: str):
        self.text = text

    def execute(self, report: "ScompLinkHTMLReport") -> "ScompLinkHTMLReport":
        report.add_text(self.text)
        return report


class GraphStep(ReportStep):
    """
    Adds a Plotly figure to the report.

    Parameters
    ----------
    fig : plotly.graph_objs.Figure
    title : str
    """

    def __init__(self, fig: Any, title: str):
        self.fig = fig
        self.title = title

    def execute(self, report: "ScompLinkHTMLReport") -> "ScompLinkHTMLReport":
        report.add_graph(self.fig, self.title)
        return report


class TableStep(ReportStep):
    """
    Adds a DataFrame table to the report.

    Parameters
    ----------
    df : pd.DataFrame
    title : str
    """

    def __init__(self, df: "pd.DataFrame", title: str):
        self.df = df
        self.title = title

    def execute(self, report: "ScompLinkHTMLReport") -> "ScompLinkHTMLReport":
        report.add_dataframe(self.df, self.title)
        return report


class RawGraphStep(ReportStep):
    """
    Adds a RAWGraphs SVG string to the report.

    Parameters
    ----------
    svg : str
        SVG markup returned by rawgraphs functions.
    title : str
    """

    def __init__(self, svg: str, title: str):
        self.svg = svg
        self.title = title

    def execute(self, report: "ScompLinkHTMLReport") -> "ScompLinkHTMLReport":
        report.add_rawgraphs_to_report(self.svg, self.title)
        return report


class HighchartsStep(ReportStep):
    """
    Adds a Highcharts chart to the report.

    Parameters
    ----------
    html_snippet : str
        HTML string returned by scomp_link.utils.highcharts functions
        (streamgraphs, calendar_heatmap, calendar_gantt, etc.).
    title : str
        Optional title shown above the chart.
    """

    def __init__(self, html_snippet: str, title: str = ""):
        self.html_snippet = html_snippet
        self.title = title

    def execute(self, report: "ScompLinkHTMLReport") -> "ScompLinkHTMLReport":
        if self.title:
            report.add_title(self.title)
        report.add_highcharts(self.html_snippet)
        return report


class CodeStep(ReportStep):
    """
    Adds a syntax-highlighted code block to the report.

    Parameters
    ----------
    code : str
        Source code to display.
    language : str
        Language for highlighting (default: "python").
    title : str
        Optional title above the code.
    output : str | None
        Optional program output shown below in terminal-style box.
    line_numbers : bool
        Show line numbers (default: False).
    collapsed : bool
        Wrap in collapsible element (default: False).
    """

    def __init__(
        self,
        code: str,
        language: str = "python",
        title: str = "",
        output: str | None = None,
        line_numbers: bool = False,
        collapsed: bool = False,
    ):
        self.code = code
        self.language = language
        self.title = title
        self.output = output
        self.line_numbers = line_numbers
        self.collapsed = collapsed

    def execute(self, report: "ScompLinkHTMLReport") -> "ScompLinkHTMLReport":
        report.add_code_block(
            self.code, self.language, self.title, self.output, line_numbers=self.line_numbers, collapsed=self.collapsed
        )
        return report


class DiffStep(ReportStep):
    """
    Adds a side-by-side diff view to the report.

    Parameters
    ----------
    old_code : str
        Original code (left side, deletions in red).
    new_code : str
        Modified code (right side, additions in green).
    language : str
        Language for highlighting (default: "python").
    title : str
        Optional title above the diff.
    old_label : str
        Label for old version (default: "before").
    new_label : str
        Label for new version (default: "after").
    collapsed : bool
        Wrap in collapsible element (default: False).
    """

    def __init__(
        self,
        old_code: str,
        new_code: str,
        language: str = "python",
        title: str = "",
        old_label: str = "before",
        new_label: str = "after",
        collapsed: bool = False,
    ):
        self.old_code = old_code
        self.new_code = new_code
        self.language = language
        self.title = title
        self.old_label = old_label
        self.new_label = new_label
        self.collapsed = collapsed

    def execute(self, report: "ScompLinkHTMLReport") -> "ScompLinkHTMLReport":
        report.add_diff(
            self.old_code,
            self.new_code,
            self.language,
            self.title,
            self.old_label,
            self.new_label,
            collapsed=self.collapsed,
        )
        return report


class SaveStep(ReportStep):
    """
    Saves the report as HTML. Automatically closes any open section before saving.

    Parameters
    ----------
    path : str
        Output file path (e.g. "report.html").

    Example
    -------
    ... >> SaveStep("output/report.html")
    """

    def __init__(self, path: str):
        self.path = path

    def execute(self, report: "ScompLinkHTMLReport") -> "ScompLinkHTMLReport":
        if report.section_just_open:
            report.close_section()
        report.save_html(self.path)
        logger.info(f"Report saved to {self.path}")
        return report


class LogStep(Step):
    """
    A transparent step that logs the current target's state without modifying it.
    Works in both ML and Report chains.

    Parameters
    ----------
    message : str | None
        Optional message prefix. Defaults to the step position label.

    Example
    -------
    CleanStep(df) >> LogStep("after clean") >> SelectStep("y") >> TrainStep("regression")
    """

    def __init__(self, message: str | None = None):
        self.message = message

    def execute(self, target: Any) -> Any:
        from scomp_link.core import ScompLinkPipeline
        from scomp_link.utils.report_html import ScompLinkHTMLReport

        label = self.message or "LogStep"
        if isinstance(target, ScompLinkPipeline):
            df_shape = target.df.shape if target.df is not None else "no data"
            logger.info(f"[{label}] pipeline — df={df_shape}, model_type={target.model_type!r}")
        elif isinstance(target, ScompLinkHTMLReport):
            logger.info(f"[{label}] report — section_open={target.section_just_open}")
        else:
            logger.info(f"[{label}] target={type(target).__name__}")
        return target


class _BoundPipelineStep(MLStep):
    """Internal step that injects an existing ScompLinkPipeline into the chain execution.
    Created automatically when using pipeline >> step syntax."""

    def __init__(self, pipeline: "ScompLinkPipeline"):
        self._pipeline = pipeline

    def execute(self, pipeline: "ScompLinkPipeline") -> "ScompLinkPipeline":
        # Ignore the auto-created pipeline, use the bound one
        return self._pipeline


__all__ = [
    "Step",
    "Chain",
    "MLStep",
    "ReportStep",
    "CleanStep",
    "SelectStep",
    "ModelStep",
    "TrainStep",
    "SectionStep",
    "TitleStep",
    "TextStep",
    "GraphStep",
    "TableStep",
    "RawGraphStep",
    "HighchartsStep",
    "SubtitleStep",
    "SaveStep",
    "LogStep",
]
