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

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import date, datetime
from plotly.subplots import make_subplots
from scomp_link.utils.report_html import ScompLinkHTMLReport

from scomp_link.utils.logger import get_logger
logger = get_logger(__name__)


from scomp_link.utils.colors import PRIMARY, LIGHT, MEDIUM_LIGHT, MEDIUM, MEDIUM_DARK, DARK, DARKEST


def multiple_histograms(variable_float_for_distribution,
                        category_variable,
                        category_name='x_label',
                        y_label='y_label',
                        h=300):
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
        logger.info('\x1b[0;37;42m Correct import of varibles \x1b[0m')
    except:
        logger.info('\x1b[0;37;41m Error in importing variables \x1b[0m')

    # find categories and split the dataset
    try:
        labels = np.unique(categ)
        sizes = np.asarray([len(np.where(categ == i)[0]) for i in labels]).astype(int)
        location = [np.where(categ == i)[0] for i in labels]
        logger.info('\x1b[0;37;42m Correct categorisation of the dataset \x1b[0m')
    except:
        logger.info('\x1b[0;37;41m Error in the categorisation of the dataset \x1b[0m')

    # Troppe categorie
    if len(sizes)>10:
        logger.info('\x1b[0;37;41m Ended because there are too many categories \x1b[0m')
        return None


    try:
        fig = make_subplots(rows=len(location), cols=1)
        logger.info('\x1b[0;37;42m Correct initialisation of the image \x1b[0m')
    except:
        logger.info('\x1b[0;37;41m Image initialisation error \x1b[0m')

    # creo grafici
    for i, q in enumerate(location):
        if sizes[i] < 5:
            logger.info('\x1b[0;37;41m Category ' + labels[i] + ' with less than 5 values, will not be printed \x1b[0m')
        else:
            values = num[q]
            arrayhist = np.histogram(values)
            spaziodist = (arrayhist[1][1] - arrayhist[1][0]) / 2
            ls_var = []
            ls_mean = []
            ls_count = []
            for bin in arrayhist[1][:-1]:
                x_ = [i for i in list(values) if i >= bin-spaziodist and i<bin+spaziodist]
                ls_var.append(np.var(x_))
                ls_mean.append(np.mean(x_))
                ls_count.append(len(x_))
            fig.add_trace(go.Histogram(x = values, nbinsx = len(arrayhist[1]) - 1,
                                        marker_color = MEDIUM_LIGHT[i],
                                        legendgroup=f"group{i+1}",
                                        legendgrouptitle_text=labels[i],
                                        name='Histogram',
                                        opacity = 0.9), row = i+1, col = 1)
            fig.add_shape(type = 'line', x0 = values.mean(), y0 = 0, x1 = values.mean(), y1 = arrayhist[0].max(),
                          line = dict(color = DARKEST[i], width = 2, dash = 'dot'), row = i+1, col = 1)
            fig.add_trace(go.Scatter(x = [round(i, 3) for i in ls_mean], y = ls_count,
                                     mode = 'markers',
                                     error_x = dict(
                                         type = 'data',
                                         symmetric = True,
                                         array = [round(i, 3) for i in np.sqrt(ls_var)],
                                         thickness = 1.5,
                                         width = 2,
                                         color = DARK[i]
                                     ),
                                     marker = dict(size = 4),
                                     marker_color = PRIMARY[i],
                                     legendgroup=f"group{i+1}",
                                     name = 'Standard deviation'), row = i+1, col = 1)
            fig.add_trace(go.Scatter(x = arrayhist[1][:-1],
                                     y = (np.cumsum(arrayhist[0]) / max(np.cumsum(arrayhist[0])) * max(arrayhist[0])),
                                     mode = 'lines',
                                     legendgroup=f"group{i+1}",
                                     name = 'Cumulative function', line_color=PRIMARY[i]), row = i+1, col = 1)
            logger.info('\x1b[0;37;42m Category ' + labels[i] + ' done! \x1b[0m')


    style = {f'yaxis{round(len(location)/2) if round(len(location)/2)>0 else ""}_title': y_label,
             f'xaxis{len(location) if round(len(location)/2)>0 else ""}_title': str(category_name)}
    fig.update_layout(barmode = 'overlay',
                      legend = dict(
                                    orientation="h",
                                    yanchor="bottom",
                                    y=1.02,
                                    xanchor="right",
                                    x=1,
                                    font = dict(size = 16)),
                      height=h*len(location),
                      **style)
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
    return multiple_histograms(variable_float_for_distribution,
                        [name_of_the_column]*len(variable_float_for_distribution),
                        category_name=name_of_the_column,
                        h=h)


def barchart(categories, metric_values_list, x_axis_title='Category', y_axis_titles=None, order='asc', categorysorted=None, metric_values_line_list=None, y_line_axis_titles=None, percentage_y = True):
    if not categorysorted:
        categorysorted = categories

    if order == 'asc':
        sorted_indices = sorted(range(len(categories)), key=lambda i: categorysorted.index(categories[i]))
    else:
        sorted_indices = sorted(range(len(categories)), key=lambda i: categorysorted.index(categories[i]), reverse=True)

    sorted_categories = [categories[i] for i in sorted_indices]
    list_tmp=[]
    for l in metric_values_list:
        list_tmp.append([l[i] for i in sorted_indices])
    metric_values_list = list_tmp
    if type(metric_values_line_list)== list:
        list_tmp=[]
        for l in metric_values_line_list:
            list_tmp.append([l[i] for i in sorted_indices])
        metric_values_line_list = list_tmp


    num_subplots = len(metric_values_list)

    fig = make_subplots(rows=num_subplots, cols=1, shared_xaxes=True, vertical_spacing=0.05)

    if isinstance(y_axis_titles, str):
        y_axis_titles = [y_axis_titles] * num_subplots  # Ripeti la stessa stringa per tutti i titoli delle barre

    if not y_line_axis_titles:
        y_line_axis_titles = [None] * num_subplots
    elif isinstance(y_line_axis_titles, str):
        y_line_axis_titles = [y_line_axis_titles] * num_subplots  # Ripeti la stessa stringa per tutti i titoli delle linee

    for i in range(num_subplots):
        trace = go.Bar(
            x=sorted_categories,
            y=metric_values_list[i],
            name=y_axis_titles[i],
            marker_color=PRIMARY[i % len(PRIMARY)],  # Cicla i colori se necessario
            hovertemplate='%{y} %{x}<extra>'+y_axis_titles[i]+'</extra>'
        )
        fig.add_trace(trace, row=i + 1, col=1)

        if metric_values_line_list and i < len(metric_values_line_list):
            line_trace = go.Scatter(
                x=sorted_categories,
                y=metric_values_line_list[i],
                mode='lines',
                name='Line',
                line=dict(color=MEDIUM_LIGHT[i % len(MEDIUM_LIGHT)]),  # Cicla i colori delle linee se necessario
                hovertemplate='%{y} %{x}<extra>'+y_line_axis_titles[i]+'</extra>'
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
        title='',
        xaxis_title=x_axis_title,
        showlegend=False,
        height=final_height  # Imposta l'altezza finale in base al numero di metriche
    )

    return fig


def area_chart(date_list, lines, title_text='Trend analysis', x_label='date', y_labels='value', format_date="%Y-%m-%d", yaxis_ticksuffix= ''):
    if format_date:
        dt = [datetime.strptime(i, format_date) for i in date_list]
    else:
        dt = date_list
    if isinstance(y_labels, str):
        y_axis_titles = [y_labels] * len(lines)
    else:
        y_axis_titles = y_labels

    plotly_axis = dict(
        showline = True,
        showgrid = False,
        showticklabels = True,
        linecolor = 'rgb(204, 204, 204)',
        linewidth = 2,
        ticks = 'outside',
        tickfont = dict(
            family = 'Arial',
            size = 12,
            color = 'rgb(82, 82, 82)'
        ))


    fig = go.Figure()
    for i, line, line_name in zip(range(len(lines)), lines, y_axis_titles):
        fig.add_trace(
            go.Scatter(x = dt, y = line, name = line_name, hoverinfo = 'x+y',  mode = 'lines',
        line = dict(width = 0.5, color = PRIMARY[i % len(PRIMARY)]),
        stackgroup = 'one'))

    fig.update_layout(
        xaxis = plotly_axis,
        yaxis = plotly_axis,
        yaxis_ticksuffix = yaxis_ticksuffix,
        legend = dict(
            orientation = "h",
            yanchor = "bottom",
            y = 1.02,
            xanchor = "right",
            x = 1,
            font = dict(size = 16)),
        height = 800,
        autosize = True,
        plot_bgcolor = 'white',
        title_text = title_text,
    )
    return fig



def linechart(date_list, lines, title_text='Trend analysis', x_label='date', y_labels='value', format_date = "%Y-%m-%d", yaxis_ticksuffix= ''):
    if format_date:
        dt = [datetime.strptime(i, format_date) for i in date_list]
    else:
        dt = date_list
    if isinstance(y_labels, str):
        y_axis_titles = [y_labels] * len(lines)  # Ripeti la stessa stringa per tutti i titoli delle barre
    else:
        y_axis_titles=y_labels
    plotly_axis = dict(
        showline = True,
        showgrid = False,
        showticklabels = True,
        linecolor = 'rgb(204, 204, 204)',
        linewidth = 2,
        ticks = 'outside',
        tickfont = dict(
            family = 'Arial',
            size = 12,
            color = 'rgb(82, 82, 82)'
        ))
    fig = go.Figure()
    for i, line, line_name in zip(range(len(lines)), lines, y_axis_titles):
        fig.add_trace(go.Scatter(x = dt, y = line, mode = "lines", name = line_name, line_color=PRIMARY[i % len(PRIMARY)]))
    fig.update_layout(
        xaxis = plotly_axis,
        yaxis = plotly_axis,
        yaxis_ticksuffix = yaxis_ticksuffix,
          legend = dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1,
                        font = dict(size = 16)),
        height = 800,
        autosize = True,
        plot_bgcolor = 'white',
        title_text = title_text,
    )
    return fig



if __name__ == '__main__':

    demo_report = ScompLinkHTMLReport('This is a demo report')


    x1 = np.random.normal(85, 3, 1000)
    x2 = [['Group A', 'Group B', 'Group C'][i] for i in np.random.randint(0, 3, 1000)]
    fig = multiple_histograms(x1, x2, 'Distribution by category')
    demo_report.add_graph_to_report(fig, 'Distribution by category')

    x1 = np.random.normal(45, 7, 1000)
    fig = histogram(x1, 'Sample age')
    demo_report.add_graph_to_report(fig, 'Sample age distribution')
    demo_report.save_html('demo_report.html')


