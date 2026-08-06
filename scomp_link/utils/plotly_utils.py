# -*- coding: utf-8 -*-
"""
██████╗ ██╗      ██████╗ ████████╗██╗  ██╗   ██╗
██╔══██╗██║     ██╔═══██╗╚══██╔══╝██║  ╚██╗ ██╔╝
██████╔╝██║     ██║   ██║   ██║   ██║   ╚████╔╝
██╔═══╝ ██║     ██║   ██║   ██║   ██║    ╚██╔╝
██║     ███████╗╚██████╔╝   ██║   ███████╗██║
╚═╝     ╚══════╝ ╚═════╝    ╚═╝   ╚══════╝╚═╝

Plotly visualization utilities for scomp-link.
Provides histogram, bar chart, line chart, and area chart functions
with consistent styling and color palettes.
"""

from datetime import date, datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scomp_link.utils.logger import get_logger
from scomp_link.utils.report_html import ScompLinkHTMLReport

logger = get_logger(__name__)


from scomp_link.utils.colors import DARK, DARKEST, LIGHT, MEDIUM, MEDIUM_DARK, MEDIUM_LIGHT, PRIMARY


def multiple_histograms(
    variable_float_for_distribution, category_variable, category_name="x_label", y_label="y_label", h=300
):
    """
    \033[95m
    \n\n
    \n Function description:\nThis function returns graphs on the distribution of a variable divided into several graphs based on a categorical variable
    \n\n
    \n Packages on which the function depends are:\n
    import numpy as np\n
    import plotly.graph_objects as go\n
    PARAMETERS:\n
     1. \t  variable_float_for_distribution: a numeric variable cleaned of missing values and None. \n
     2. \t  category_variable: for each value of which the variable above expresses a category.\n
     3. \t  category_name: name from of the characteristic expressed in the categories.\n
     4. \t  h: size of the final image
    \033[0m

    \033[96m
    Example:

    from scomp_link.utils.plotly_utils import multiple_histograms
    x1 = np.random.normal(85, 3, 1000)
    x2 = [['Group A', 'Group B', 'Group C'][i] for i in np.random.randint(0, 3, 1000)]
    fig = multiple_histograms(x1, x2, 'Distribution Comparison')
    fig.show()
    \033[0m
    """

    # trasformo le variabili in numpy array
    try:
        num = np.asarray(variable_float_for_distribution).astype(float)
        categ = np.asarray(category_variable).astype(str)
        logger.info("\x1b[0;37;42m Correct import of varibles \x1b[0m")
    except:
        logger.info("\x1b[0;37;41m Error in importing variables \x1b[0m")

    # find categories and split the dataset
    try:
        labels = np.unique(categ)
        sizes = np.asarray([len(np.where(categ == i)[0]) for i in labels]).astype(int)
        location = [np.where(categ == i)[0] for i in labels]
        logger.info("\x1b[0;37;42m Correct categorisation of the dataset \x1b[0m")
    except:
        logger.info("\x1b[0;37;41m Error in the categorisation of the dataset \x1b[0m")

    # Troppe categorie
    if len(sizes) > 10:
        logger.info("\x1b[0;37;41m Ended because there are too many categories \x1b[0m")
        return None

    try:
        fig = make_subplots(rows=len(location), cols=1)
        logger.info("\x1b[0;37;42m Correct initialisation of the image \x1b[0m")
    except:
        logger.info("\x1b[0;37;41m Image initialisation error \x1b[0m")

    # creo grafici
    for i, q in enumerate(location):
        if sizes[i] < 5:
            logger.info("\x1b[0;37;41m Category " + labels[i] + " with less than 5 values, will not be printed \x1b[0m")
        else:
            values = num[q]
            arrayhist = np.histogram(values)
            spaziodist = (arrayhist[1][1] - arrayhist[1][0]) / 2
            ls_var = []
            ls_mean = []
            ls_count = []
            for bin in arrayhist[1][:-1]:
                x_ = [i for i in list(values) if i >= bin - spaziodist and i < bin + spaziodist]
                ls_var.append(np.var(x_))
                ls_mean.append(np.mean(x_))
                ls_count.append(len(x_))
            fig.add_trace(
                go.Histogram(
                    x=values,
                    nbinsx=len(arrayhist[1]) - 1,
                    marker_color=MEDIUM_LIGHT[i],
                    legendgroup=f"group{i+1}",
                    legendgrouptitle_text=labels[i],
                    name="Histogram",
                    opacity=0.9,
                ),
                row=i + 1,
                col=1,
            )
            fig.add_shape(
                type="line",
                x0=values.mean(),
                y0=0,
                x1=values.mean(),
                y1=arrayhist[0].max(),
                line=dict(color=DARKEST[i], width=2, dash="dot"),
                row=i + 1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=[round(i, 3) for i in ls_mean],
                    y=ls_count,
                    mode="markers",
                    error_x=dict(
                        type="data",
                        symmetric=True,
                        array=[round(i, 3) for i in np.sqrt(ls_var)],
                        thickness=1.5,
                        width=2,
                        color=DARK[i],
                    ),
                    marker=dict(size=4),
                    marker_color=PRIMARY[i],
                    legendgroup=f"group{i+1}",
                    name="Standard deviation",
                ),
                row=i + 1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=arrayhist[1][:-1],
                    y=(np.cumsum(arrayhist[0]) / max(np.cumsum(arrayhist[0])) * max(arrayhist[0])),
                    mode="lines",
                    legendgroup=f"group{i+1}",
                    name="Cumulative function",
                    line_color=PRIMARY[i],
                ),
                row=i + 1,
                col=1,
            )
            logger.info("\x1b[0;37;42m Category " + labels[i] + " done! \x1b[0m")

    style = {
        f'yaxis{round(len(location)/2) if round(len(location)/2)>0 else ""}_title': y_label,
        f'xaxis{len(location) if round(len(location)/2)>0 else ""}_title': str(category_name),
    }
    fig.update_layout(
        barmode="overlay",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=16)),
        height=h * len(location),
        **style,  # type: ignore[arg-type]
    )
    return fig


def histogram(variable_float_for_distribution, name_of_the_column, h=600):
    """
    \033[95m
    \n\n
    \n Function description:\nThis function returns graphs on the distribution of a variable divided into several graphs based on a categorical variable
    \n\n
    \n Packages on which the function depends are:\n
    import numpy as np\n
    import plotly.graph_objects as go\n
    PARAMETERS:\n
     1. \t  variable_float_for_distribution: a numeric variable cleaned of missing values and None. \n
     2. \t  category_name: name from of the characteristic expressed in the categories.\n
     3. \t  h: size of the final image
    \033[0m

    \033[96m
    Example:

    from scomp_link.utils.plotly_utils import histogram
    x1 = np.random.normal(45, 3, 1000)
    fig = histogram(x1, 'Sample age')
    fig.show()
    \033[0m
    """
    return multiple_histograms(
        variable_float_for_distribution,
        [name_of_the_column] * len(variable_float_for_distribution),
        category_name=name_of_the_column,
        h=h,
    )


def barchart(
    categories,
    metric_values_list,
    x_axis_title="Category",
    y_axis_titles=None,
    order="asc",
    categorysorted=None,
    metric_values_line_list=None,
    y_line_axis_titles=None,
    percentage_y=True,
):
    if not categorysorted:
        categorysorted = categories

    if order == "asc":
        sorted_indices = sorted(range(len(categories)), key=lambda i: categorysorted.index(categories[i]))
    else:
        sorted_indices = sorted(range(len(categories)), key=lambda i: categorysorted.index(categories[i]), reverse=True)

    sorted_categories = [categories[i] for i in sorted_indices]
    list_tmp = []
    for l in metric_values_list:
        list_tmp.append([l[i] for i in sorted_indices])
    metric_values_list = list_tmp
    if type(metric_values_line_list) == list:
        list_tmp = []
        for l in metric_values_line_list:
            list_tmp.append([l[i] for i in sorted_indices])
        metric_values_line_list = list_tmp

    num_subplots = len(metric_values_list)

    fig = make_subplots(rows=num_subplots, cols=1, shared_xaxes=True, vertical_spacing=0.05)

    if isinstance(y_axis_titles, str):
        y_axis_titles = [y_axis_titles] * num_subplots  # Ripeti la stessa stringa per tutti i titoli delle barre
    elif y_axis_titles is None:
        y_axis_titles = [f"Series {i+1}" for i in range(num_subplots)]

    if not y_line_axis_titles:
        y_line_axis_titles = [""] * num_subplots
    elif isinstance(y_line_axis_titles, str):
        y_line_axis_titles = [
            y_line_axis_titles
        ] * num_subplots  # Ripeti la stessa stringa per tutti i titoli delle linee

    for i in range(num_subplots):
        trace = go.Bar(
            x=sorted_categories,
            y=metric_values_list[i],
            name=y_axis_titles[i],
            marker_color=PRIMARY[i % len(PRIMARY)],  # Cicla i colori se necessario
            hovertemplate="%{y} %{x}<extra>" + y_axis_titles[i] + "</extra>",
        )
        fig.add_trace(trace, row=i + 1, col=1)

        if metric_values_line_list and i < len(metric_values_line_list):
            line_trace = go.Scatter(
                x=sorted_categories,
                y=metric_values_line_list[i],
                mode="lines",
                name="Line",
                line=dict(color=MEDIUM_LIGHT[i % len(MEDIUM_LIGHT)]),  # Cicla i colori delle linee se necessario
                hovertemplate="%{y} %{x}<extra>" + y_line_axis_titles[i] + "</extra>",
            )
            fig.add_trace(line_trace, row=i + 1, col=1)

    for i in range(num_subplots):
        if percentage_y:
            fig.update_yaxes(title_text=y_axis_titles[i], row=i + 1, col=1, tickformat=".2%")
        else:
            fig.update_yaxes(title_text=y_axis_titles[i], row=i + 1, col=1)

    # Calcola l'altezza dinamica in base al numero di metriche
    base_height = 400
    progressive_factor = 100  # Modifica questo fattore a tuo piacimento
    final_height = base_height + (num_subplots - 1) * progressive_factor

    fig.update_layout(
        title="",
        xaxis_title=x_axis_title,
        showlegend=False,
        height=final_height,  # Imposta l'altezza finale in base al numero di metriche
    )

    return fig


def area_chart(
    date_list,
    lines,
    title_text="Trend analysis",
    x_label="date",
    y_labels: "str | list[str]" = "value",
    format_date="%Y-%m-%d",
    yaxis_ticksuffix="",
):
    if format_date:
        dt = [datetime.strptime(i, format_date) for i in date_list]
    else:
        dt = date_list
    if isinstance(y_labels, str):
        y_axis_titles = [y_labels] * len(lines)
    else:
        y_axis_titles = y_labels

    plotly_axis = dict(
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor="rgb(204, 204, 204)",
        linewidth=2,
        ticks="outside",
        tickfont=dict(family="Arial", size=12, color="rgb(82, 82, 82)"),
    )

    fig = go.Figure()
    for i, line, line_name in zip(range(len(lines)), lines, y_axis_titles):
        fig.add_trace(
            go.Scatter(
                x=dt,
                y=line,
                name=line_name,
                hoverinfo="x+y",
                mode="lines",
                line=dict(width=0.5, color=PRIMARY[i % len(PRIMARY)]),
                stackgroup="one",
            )
        )

    fig.update_layout(
        xaxis=plotly_axis,
        yaxis=plotly_axis,
        yaxis_ticksuffix=yaxis_ticksuffix,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=16)),
        height=800,
        autosize=True,
        plot_bgcolor="white",
        title_text=title_text,
    )
    return fig


def linechart(
    date_list,
    lines,
    title_text="Trend analysis",
    x_label="date",
    y_labels: "str | list[str]" = "value",
    format_date="%Y-%m-%d",
    yaxis_ticksuffix="",
):
    if format_date:
        dt = [datetime.strptime(i, format_date) for i in date_list]
    else:
        dt = date_list
    if isinstance(y_labels, str):
        y_axis_titles = [y_labels] * len(lines)  # Ripeti la stessa stringa per tutti i titoli delle barre
    else:
        y_axis_titles = y_labels
    plotly_axis = dict(
        showline=True,
        showgrid=False,
        showticklabels=True,
        linecolor="rgb(204, 204, 204)",
        linewidth=2,
        ticks="outside",
        tickfont=dict(family="Arial", size=12, color="rgb(82, 82, 82)"),
    )
    fig = go.Figure()
    for i, line, line_name in zip(range(len(lines)), lines, y_axis_titles):
        fig.add_trace(go.Scatter(x=dt, y=line, mode="lines", name=line_name, line_color=PRIMARY[i % len(PRIMARY)]))
    fig.update_layout(
        xaxis=plotly_axis,
        yaxis=plotly_axis,
        yaxis_ticksuffix=yaxis_ticksuffix,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=16)),
        height=800,
        autosize=True,
        plot_bgcolor="white",
        title_text=title_text,
    )
    return fig


def fill_timeslots(values, n_slots: int, fill_value: float = 0.0) -> np.ndarray:
    """Ensure array has exactly n_slots elements, padding or truncating as needed.

    Useful for preparing time-series data where some slots may be missing.

    :param values: list or array of numeric values
    :param n_slots: desired output length
    :param fill_value: value used to pad if input is shorter (default 0.0)
    :return: numpy array of length n_slots

    ## example
    from scomp_link.utils.plotly_utils import fill_timeslots
    result = fill_timeslots([1, 2, 3], n_slots=5)  # [1, 2, 3, 0, 0]
    """
    arr = np.asarray(values, dtype=float)
    if len(arr) >= n_slots:
        return arr[:n_slots]
    padded = np.full(n_slots, fill_value, dtype=float)
    padded[:len(arr)] = arr
    return padded


def normalize_to_index(values, baseline: float = 100.0) -> np.ndarray:
    """Normalize array so that its mean equals the baseline value.

    Useful for creating index charts where baseline=100 represents the average.
    Returns zeros if the input mean is zero.

    :param values: list or array of numeric values
    :param baseline: target mean value (default 100.0)
    :return: numpy array with mean approximately equal to baseline

    ## example
    from scomp_link.utils.plotly_utils import normalize_to_index
    result = normalize_to_index([2, 4, 6], baseline=100)  # [50, 100, 150]
    """
    arr = np.asarray(values, dtype=float)
    mean_val = arr.mean()
    if mean_val == 0:
        return np.zeros_like(arr)
    return arr / mean_val * baseline


def index_chart(
    series_dict: dict,
    x_labels: list[str],
    title: str,
    baseline: float = 100.0,
    height: int = 400,
    colors: list[str] | None = None,
) -> go.Figure:
    """Create an index chart with toggle buttons to switch between series groups.

    Each series is normalized so its mean equals `baseline` (default 100).
    A horizontal reference line is drawn at the baseline.

    series_dict formats:
    - Simple: {"A": [values], "B": [values]} — one solid trace per key
    - Paired: {"A": {"solid": [values], "dashed": [values]}} — two traces per key

    :param series_dict: dict mapping series names to values or {"solid": ..., "dashed": ...}
    :param x_labels: list of x-axis labels
    :param title: chart title
    :param baseline: index baseline value (default 100)
    :param height: chart height in pixels (default 400)
    :param colors: optional list of colors for series (cycles if fewer than series)
    :return: plotly Figure

    ## example
    from scomp_link.utils.plotly_utils import index_chart
    fig = index_chart(
        {"Group A": {"solid": [2,4,3,5], "dashed": [3,3,4,4]},
         "Group B": {"solid": [5,3,4,2], "dashed": [4,4,3,3]}},
        x_labels=["Q1", "Q2", "Q3", "Q4"],
        title="Performance Index"
    )
    """
    default_colors = [c for group in [PRIMARY, MEDIUM, DARK, MEDIUM_LIGHT, MEDIUM_DARK, DARKEST, LIGHT] for c in group]
    palette = colors or default_colors

    fig = go.Figure()

    # Determine format
    first_val = next(iter(series_dict.values()))
    is_paired = isinstance(first_val, dict)

    group_names = list(series_dict.keys())
    traces_per_group = 2 if is_paired else 1

    for i, (name, data) in enumerate(series_dict.items()):
        color = palette[i % len(palette)]
        visible = (i == 0)

        if is_paired:
            solid_vals = normalize_to_index(data["solid"], baseline)
            dashed_vals = normalize_to_index(data["dashed"], baseline)
            fig.add_trace(go.Scatter(
                x=x_labels, y=solid_vals.tolist(),
                name=f"{name} (solid)", line=dict(color=color, width=3),
                mode="lines", visible=visible,
            ))
            fig.add_trace(go.Scatter(
                x=x_labels, y=dashed_vals.tolist(),
                name=f"{name} (dashed)", line=dict(color=color, width=3, dash="dash"),
                mode="lines", visible=visible,
            ))
        else:
            vals = normalize_to_index(data, baseline)
            fig.add_trace(go.Scatter(
                x=x_labels, y=vals.tolist(),
                name=name, line=dict(color=color, width=3),
                mode="lines", visible=visible,
            ))

    fig.add_hline(y=baseline, line_dash="dot", line_color="rgba(128,128,128,0.4)")

    # Toggle buttons
    total_traces = len(group_names) * traces_per_group
    buttons = []
    for i, name in enumerate(group_names):
        vis = [False] * total_traces
        for j in range(traces_per_group):
            vis[i * traces_per_group + j] = True
        buttons.append(dict(
            label=f"  {name}  ", method="update",
            args=[{"visible": vis}, {"title": f"{title} — {name}"}],
        ))

    # "All" button
    buttons.append(dict(
        label="  All  ", method="update",
        args=[{"visible": [True] * total_traces}, {"title": title}],
    ))

    fig.update_layout(
        title=f"{title} — {group_names[0]}" if group_names else title,
        height=height,
        xaxis_title="", yaxis_title=f"Index ({baseline} = average)",
        xaxis=dict(tickangle=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
        updatemenus=[dict(
            type="buttons", direction="right",
            x=0.5, xanchor="center", y=-0.2, yanchor="top",
            buttons=buttons,
        )],
        margin=dict(b=100),
    )
    return fig


def stacked_area_comparison(
    data_left: dict[str, list[float]],
    data_right: dict[str, list[float]],
    categories: list[str],
    x_labels: list[str],
    title: str,
    subplot_titles: tuple[str, str] = ("Left", "Right"),
    height: int = 400,
    colors: list[str] | None = None,
) -> go.Figure:
    """Create side-by-side 100%% stacked area charts for comparison.

    Each slot sums to 100%%. Useful for comparing composition between two sources.

    :param data_left: dict mapping category names to value lists for left subplot
    :param data_right: dict mapping category names to value lists for right subplot
    :param categories: ordered list of category names (determines stacking order and legend)
    :param x_labels: x-axis labels
    :param title: overall chart title
    :param subplot_titles: tuple of (left_title, right_title)
    :param height: chart height in pixels (default 400)
    :param colors: optional list of colors for categories
    :return: plotly Figure with 2 subplots

    ## example
    from scomp_link.utils.plotly_utils import stacked_area_comparison
    fig = stacked_area_comparison(
        data_left={"Young": [30,25,20], "Mid": [40,45,50], "Senior": [30,30,30]},
        data_right={"Young": [20,20,25], "Mid": [50,50,45], "Senior": [30,30,30]},
        categories=["Young", "Mid", "Senior"],
        x_labels=["Morning", "Afternoon", "Evening"],
        title="Age Composition",
        subplot_titles=("Source A", "Source B")
    )
    """
    default_colors = [c for group in [PRIMARY, MEDIUM, DARK, MEDIUM_LIGHT, MEDIUM_DARK, DARKEST, LIGHT] for c in group]
    palette = colors or default_colors

    def _normalize_to_pct(data_dict: dict, cats: list[str], n_points: int) -> dict[str, list[float]]:
        """Normalize each slot to sum to 100%."""
        result = {}
        for cat in cats:
            result[cat] = list(data_dict.get(cat, [0.0] * n_points))
        # Normalize each position
        for j in range(n_points):
            total = sum(result[cat][j] for cat in cats)
            if total > 0:
                for cat in cats:
                    result[cat][j] = result[cat][j] / total * 100
            else:
                for cat in cats:
                    result[cat][j] = 0.0
        return result

    n_points = len(x_labels)
    left_pct = _normalize_to_pct(data_left, categories, n_points)
    right_pct = _normalize_to_pct(data_right, categories, n_points)

    fig = make_subplots(rows=1, cols=2, subplot_titles=list(subplot_titles), horizontal_spacing=0.05)

    for data_pct, col_idx in [(left_pct, 1), (right_pct, 2)]:
        for i, cat in enumerate(categories):
            color = palette[i % len(palette)]
            fig.add_trace(go.Scatter(
                x=x_labels, y=data_pct[cat],
                name=cat if col_idx == 1 else None,
                legendgroup=cat, showlegend=(col_idx == 1),
                stackgroup="one",
                fillcolor=color,
                line=dict(width=0.5, color=color),
                mode="lines",
            ), row=1, col=col_idx)

    fig.update_layout(
        title=title, height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5),
    )
    fig.update_xaxes(tickangle=45)
    fig.update_yaxes(title_text="% of slot", range=[0, 100])
    return fig


if __name__ == "__main__":
    demo_report = ScompLinkHTMLReport("This is a demo report")

    x1 = np.random.normal(85, 3, 1000)
    x2 = [["Group A", "Group B", "Group C"][i] for i in np.random.randint(0, 3, 1000)]
    fig = multiple_histograms(x1, x2, "Distribution by category")
    assert fig is not None
    demo_report.add_graph_to_report(fig, "Distribution by category")

    x1 = np.random.normal(45, 7, 1000)
    fig = histogram(x1, "Sample age")
    assert fig is not None
    demo_report.add_graph_to_report(fig, "Sample age distribution")
    demo_report.save_html("demo_report.html")
