# -*- coding: utf-8 -*-
"""
██╗  ██╗ ██████╗██╗  ██╗ ██████╗██╗  ██╗ █████╗ ██████╗ ████████╗ ██████╗
██║  ██║██╔════╝██║  ██║██╔════╝██║  ██║██╔══██╗██╔══██╗╚══██╔══╝██╔════╝
███████║██║     ███████║██║     ███████║███████║██████╔╝   ██║   ╚██████╗
██╔══██║██║     ██╔══██║██║     ██╔══██║██╔══██║██╔═══╝    ██║    ╚════██║
██║  ██║╚██████╗██║  ██║╚██████╗██║  ██║██║  ██║██║        ██║   ██████╔╝
╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚═╝        ╚═╝   ╚═════╝

Highcharts visualization utilities for scomp-link.
Provides streamgraph, calendar heatmap, and Gantt chart functions
for embedding in HTML reports.
"""

css = '''
    <style>
    #container {
        height: 600px;
    }

    .highcharts-figure,
    .highcharts-data-table table {
        min-width: 310px;
        max-width: 800px;
        overflow: auto;
        margin: 1em auto;
    }

    .highcharts-data-table table {
        font-family: Verdana, sans-serif;
        border-collapse: collapse;
        border: 1px solid #ebebeb;
        margin: 10px auto;
        text-align: center;
        width: 100%;
        height: auto;
        max-width: 90000px;
    }
    
    .highcharts-figure {width: 100%;
        height: auto;
        max-width: 90000px;}

    .highcharts-data-table caption {
        padding: 1em 0;
        font-size: 1.2em;
        color: #555;
    }

    .highcharts-data-table th {
        font-weight: 600;
        padding: 0.5em;
    }

    .highcharts-data-table td,
    .highcharts-data-table th,
    .highcharts-data-table caption {
        padding: 0.5em;
    }

    .highcharts-data-table thead tr,
    .highcharts-data-table tr:nth-child(even) {
        background: #f8f8f8;
    }

    .highcharts-data-table tr:hover {
        background: #f1f7ff;
    }

    </style>
    '''
js = '''
    <script src="https://code.highcharts.com/highcharts.js"></script>
    <script src="https://code.highcharts.com/modules/streamgraph.js"></script>
    <script src="https://code.highcharts.com/modules/series-label.js"></script>
    <script src="https://code.highcharts.com/modules/annotations.js"></script>
    <script src="https://code.highcharts.com/modules/exporting.js"></script>
    <script src="https://code.highcharts.com/modules/export-data.js"></script>
    <script>const colors = Highcharts.getOptions().colors;</script>
    '''



def streamgraphs(title, dates, series_dict:'{serie_name: list_values}', annotation:'dict {annotation_description: int(dates_index)}' = None, area=True):

    id_name = title.lower().replace(" ", "_")
    for p in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~':
        id_name = id_name.replace(p, '_')
    html_figure = f"""
    <figure class="highcharts-figure">
        <div id="{id_name}"></div>
    </figure>"""
    import random
    list_colors = [f'Highcharts.color(colors[{str(int(round(9*i/len(series_dict.keys()),0)))}]).brighten({str(round(i/(len(series_dict.keys())*2),1))}).get()' for i in range(len(series_dict.keys()))]
    random.shuffle(list_colors)
    colors = ', '.join(list_colors)
    annotations_text = ''
    if annotation:
        for k, v in annotation.items():
            separ = ',' if annotations_text != '' else ''
            annotations_text += separ+"""{
                           point: {
                                x: """ + str(int(v)) + """,
                                xAxis: 0,
                                y: 0,
                                yAxis: 0
                            },
                            text: '""" + str(k) + """'
                        }"""

    if area:
        type_graph = 'area'
        visibility_y = 'true'
    else:
        type_graph = 'streamgraph'
        visibility_y = 'false'
    graph = """
    <script>
    Highcharts.chart('""" + id_name + """', {

        chart: {
            type: '"""+type_graph+"""',
            marginBottom: 30,
            zoomType: 'x'
        },
        colors: [""" + colors + """],
       title: {
            floating: true,
            align: 'left',
            text: '""" + title + """'
        },
        subtitle: {
            floating: true,
            align: 'left',
            y: 30,
            text: '"""+type_graph+"""'
        },

        xAxis: {
            maxPadding: 0,
            type: 'category',
            crosshair: true,
            categories: [
    """ + "'" + "', '".join(dates) + "'" + \
               """
               ],
                       labels: {
                           align: 'left',
                           reserveSpace: false,
                           rotation: 270
                       },
                       lineWidth: 0,
                       margin: 20,
                       tickWidth: 0
                   },

                   yAxis: {
                       visible: """+visibility_y+""",
                       startOnTick: """+visibility_y+""",
                       endOnTick: """+visibility_y+"""
                   },

                   legend: {
                       enabled: false
                   },

                   annotations: [{labels: [""" + str(annotations_text) + """],
            labelOptions: {
                backgroundColor: 'rgba(255,255,255,0.5)',
                borderColor: 'silver'
            }
        }],

        plotOptions: {
            series: {
                label: {
                    minFontSize: 5,
                    maxFontSize: 15,
                    style: {
                        color: 'rgba(255,255,255,0.75)'
                    }
                },
                accessibility: {
                    exposeAsGroupOnly: true
                }
            }
        },

        // Data parsed with olympic-medals.node.js
        series: [
    """ + ", ".join(["{" + f"name:'{k.upper()}', data: [{','.join([str(int(i)) for i in v])}]" + "}" for k, v in series_dict.items()]) + \
               """
               ],

               exporting: {
                   sourceWidth: 800,
                   sourceHeight: 600
               }

           });

           </script>
               """

    return f"""
            {html_figure}
            {graph}
            """








def calendar_heatmap(title, series_dict: 'dict {"yyyy-mm-dd": round(percentage_value*100, 2)}', min=0, max =1):

    q = len(series_dict.keys())
    if q <= (7*5):
        qmin ='0'
    else:
        qmin = f'-{(int((q - (7*5))/7)+1)}'
    id_name = title.lower().replace(" ", "_")
    for p in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~':
        id_name = id_name.replace(p, '_')
    html_figure = f"""
    <figure class="highcharts-figure">
        <div id="{id_name}"></div>
    </figure>"""

    data_init = f'const data_{id_name} = ' + '[{' + '}, {'.join([f"date:'{k}', value:{v}" for k,v in series_dict.items()]) + '}];'
    data_init += f"\n const weekdays_{id_name} = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];"
    data_init += """
                    
                // The function takes in a dataset and calculates how many empty tiles needed
                // before and after the dataset is plotted.
                function generateChartData_"""+id_name+"""(data) {
                
                    // Calculate the starting weekday index (0-6 of the first date in the given
                    // array)
                    const firstWeekday = new Date(data[0].date).getDay(),
                        monthLength = data.length,
                        lastElement = data[monthLength - 1].date,
                        lastWeekday = new Date(lastElement).getDay(),
                        lengthOfWeek = 6,
                        emptyTilesFirst = firstWeekday,
                        chartData = [];
                
                    // Add the empty tiles before the first day of the month with null values to
                    // take up space in the chart
                    for (let emptyDay = 0; emptyDay < emptyTilesFirst; emptyDay++) {
                        chartData.push({
                            x: emptyDay,
                            y: 5,
                            value: null,
                            date: null,
                            custom: {
                                empty: true
                            }
                        });
                    }
                
                    // Loop through and populate with values and dates from the dataset
                    for (let day = 1; day <= monthLength; day++) {
                        // Get date from the given data array
                        const date = data[day - 1].date;
                        // Offset by thenumber of empty tiles
                        const xCoordinate = (emptyTilesFirst + day - 1) % 7;
                        const yCoordinate = Math.floor((firstWeekday + day - 1) / 7);
                        const id = day;
                
                        // Get the corresponding percentage value for the current day from the given array
                        const percentageValue = data[day - 1].value; 
                
                        chartData.push({
                            x: xCoordinate,
                            y: 5 - yCoordinate,
                            value: percentageValue,
                            date: new Date(date).getTime(),
                            custom: {
                                monthDay: id
                            }
                        });
                    }
                
                    // Fill in the missing values when dataset is looped through.
                    const emptyTilesLast = lengthOfWeek - lastWeekday;
                    for (let emptyDay = 1; emptyDay <= emptyTilesLast; emptyDay++) {
                        chartData.push({
                            x: (lastWeekday + emptyDay) % 7,
                            y: 0,
                            value: null,
                            date: null,
                            custom: {
                                empty: true
                            }
                        });
                    }
                    return chartData;
                }
    """
    data_init += f"const chartData_{id_name} = generateChartData_{id_name}(data_{id_name});"

    graph = """
            Highcharts.chart('"""+id_name+"""', {
                chart: {
                    type: 'heatmap'
                },
            
                title: {
                    text: '"""+ title+"""',
                    align: 'left'
                },
            
                subtitle: {
                    text: 'Percentage variation at day',
                    align: 'left'
                },
            
                accessibility: {
                    landmarkVerbosity: 'one'
                },
            
                tooltip: {
                    enabled: true,
                    outside: true,
                    zIndex: 20,
                    headerFormat: '',
                    pointFormat: '{#unless point.custom.empty}{point.date:%A, %b %e, %Y}{/unless}',
                    nullFormat: 'No data'
                },
            
                xAxis: {
                    categories: weekdays_"""+id_name+""",
                    opposite: true,
                    lineWidth: 26,
                    offset: 13,
                    lineColor: 'rgba(27, 26, 37, 0.2)',
                    labels: {
                        rotation: 0,
                        y: 20,
                        style: {
                            textTransform: 'uppercase',
                            fontWeight: 'bold'
                        }
                    },
                    accessibility: {
                        description: 'weekdays',
                        rangeDescription: 'X Axis is showing all 7 days of the week, starting with Sunday.'
                    }
                },
            
                yAxis: {
                    min: """+qmin+""",
                    max: 5,
                    accessibility: {
                        description: 'weeks'
                    },
                    visible: false
                },
            
                legend: {
                    align: 'right',
                    layout: 'vertical',
                    verticalAlign: 'middle'
                },
            
                colorAxis: {
                    min: """+str(round(min,2))+""",
                    stops: [
                        ["""+str(round(min+((max-min)*0.2),2))+""", 'lightblue'],
                        ["""+str(round(min+((max-min)*0.4),2))+""", '#CBDFC8'],
                        ["""+str(round(min+((max-min)*0.6),2))+""", '#F3E99E'],
                        ["""+str(round(min+((max-min)*0.9),2))+""", '#F9A05C']
                    ],
                    labels: {
                        format: '{value} %'
                    }
                },
            
                series: [{
                    keys: ['x', 'y', 'value', 'date', 'id'],
                    data: chartData_"""+id_name+""",
                    nullColor: 'rgba(196, 196, 196, 0.2)',
                    borderWidth: 2,
                    borderColor: 'rgba(196, 196, 196, 0.2)',
                    dataLabels: [{
                        enabled: true,
                        format: '{#unless point.custom.empty}{point.value:.1f}%{/unless}',
                        style: {
                            textOutline: 'none',
                            fontWeight: 'normal',
                            fontSize: '1rem'
                        }
                    }, {
                        enabled: true,
                        align: 'left',
                        verticalAlign: 'top',
                        format: '{#unless point.custom.empty}{point.custom.monthDay}{/unless}',
                        backgroundColor: 'whitesmoke',
                        padding: 2,
                        style: {
                            textOutline: 'none',
                            color: 'rgba(70, 70, 92, 1)',
                            fontSize: '0.8rem',
                            fontWeight: 'bold',
                            opacity: 0.5
                        },
                        x: 1,
                        y: 1
                    }]
                }]
            });    
    """

    return f"""
            {html_figure}
            <script>
                {data_init}
                {graph}
            </script>
            """


def calendar_gantt(title, series_dict: 'list of dict all info see https://www.highcharts.com/demo/gantt/project-management?redirect-to-jsfiddle', min_date:"yyyy-mm-dd", max_date:"yyyy-mm-dd", colors:str = None):
    if not colors:
        colors = '["#6E37FA", "#32BBB9", "#FF9408", "#F40953", "#FA32A0", "#B30095", "#FFD500", "#AAF564", "#50E6AA", "#2765F0"]'
    id_name = title.lower().replace(" ", "_")
    for p in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~':
        id_name = id_name.replace(p, '_')
    html_figure = f"""
    <figure class="highcharts-figure">
        <div id="{id_name}"></div>
    </figure>"""

    data_init = """
            const options_"""+id_name+""" = {
                chart: {
                    plotBackgroundColor: 'rgba(128,128,128,0.02)',
                    plotBorderColor: 'rgba(128,128,128,0.1)',
                    plotBorderWidth: 1
                },
            
                plotOptions: {
                    series: {
                        borderRadius: '50%',
                        connectors: {
                            dashStyle: 'ShortDot',
                            lineWidth: 2,
                            radius: 5,
                            startMarker: {
                                enabled: false
                            }
                        },
                        groupPadding: 0,
                        dataLabels: [{
                            enabled: true,
                            align: 'left',
                            format: '{point.name}',
                            padding: 10,
                            style: {
                                fontWeight: 'normal',
                                textOutline: 'none'
                            }
                        }, {
                            enabled: true,
                            align: 'right',
                            format: '{#if point.completed}{(multiply point.completed.amount 100):.0f}%{/if}',
                            padding: 10,
                            style: {
                                fontWeight: 'normal',
                                textOutline: 'none',
                                opacity: 0.6
                            }
                        }]
                    }
                },
                colors: """+colors+""",
                series: [
    """
    no_string_c = ['start','end','milestone', 'completed', 'collapsed']
    data_init += '{'+ '}, {'.join([f"name: '{components['name']}', data: [" + '{'+ '}, {'.join([', '.join([f"{k}: '{v}'" if k not in no_string_c else f"{k}: {v}" for k, v in component.items()]) for component in components['data']]) + '}' + ']' for components in series_dict]) + '}'
    data_init += """
    
                ],
             tooltip: {
                    pointFormat: '<span style="font-weight: bold">{point.name}</span><br>' +
                        '{point.start:%e %b}' +
                        '{#unless point.milestone} → {point.end:%e %b}{/unless}' +
                        '<br>' +
                        '{#if point.completed}' +
                        'Completed: {multiply point.completed.amount 100}%<br>' +
                        '{/if}' 
                },
                title: {
                    text: '"""+title+"""'
                },
                xAxis: [{
                    currentDateIndicator: {
                        color: '#2caffe',
                        dashStyle: 'ShortDot',
                        width: 2,
                        label: {
                            format: ''
                        }
                    },
                    dateTimeLabelFormats: {
                        day: '%e<br><span style="opacity: 0.5; font-size: 0.7em">%a</span>'
                    },
                    grid: {
                        borderWidth: 0
                    },
                    gridLineWidth: 1,
                    min: new Date('"""+min_date+"""').getTime(),
                    max: new Date('"""+max_date+"""').getTime()+160000000,
                    custom: {
                        weekendPlotBands: true
                    }
                }],
                yAxis: {
                    grid: {
                        borderWidth: 0
                    },
                    gridLineWidth: 0,
                    labels: {
                        symbol: {
                            width: 8,
                            height: 6,
                            x: -4,
                            y: -2
                        }
                    },
                    staticScale: 30
                },
                accessibility: {
                    keyboardNavigation: {
                        seriesNavigation: {
                            mode: 'serialize'
                        }
                    },
                    point: {
                        descriptionFormatter: function (point) {
                            const completedValue = point.completed ?
                                    point.completed.amount || point.completed : null,
                                completed = completedValue ?
                                    ' Task ' + Math.round(completedValue * 1000) / 10 + '% completed.' :
                                    '',
                                dependency = point.dependency &&
                                    point.series.chart.get(point.dependency).name,
                                dependsOn = dependency ? ' Depends on ' + dependency + '.' : '';
            
                            return Highcharts.format(
                                point.milestone ?
                                    '{point.yCategory}. Milestone at {point.x:%Y-%m-%d}. Owner: {point.owner}.{dependsOn}' :
                                    '{point.yCategory}.{completed} Start {point.x:%Y-%m-%d}, end {point.x2:%Y-%m-%d}. Owner: {point.owner}.{dependsOn}',
                                { point, completed, dependsOn }
                            );
                        }
                    }
                },

                navigator: {
                    enabled: true,
                    liveRedraw: true,
                    series: {
                        type: 'gantt',
                        pointPlacement: 0.5,
                        pointPadding: 0.25,
                        accessibility: {
                            enabled: false
                        }
                    },
                    yAxis: {
                        min: 0,
                        max: 3,
                        reversed: true,
                        categories: []
                    }
                },
            
                scrollbar: {
                    enabled: true
                },
            
                rangeSelector: {
                    enabled: true,
                    selected: 5
                },

                lang: {
                    accessibility: {
                        axis: {
                            xAxisDescriptionPlural: 'The chart has a two-part X axis showing time in both week numbers and days.'
                        }
                    }
                }
            };

    """

    graph = """
            // Plug-in to render plot bands for the weekends
            HighchartsGantt.addEvent(HighchartsGantt.Axis, 'foundExtremes', e => {
                if (e.target.options.custom && e.target.options.custom.weekendPlotBands) {
                    const axis = e.target,
                        chart = axis.chart,
                        day = 24 * 36e5,
                        isWeekend = t => /[06]/.test(chart.time.dateFormat('%w', t)),
                        plotBands = [];
            
                    let inWeekend = false;
            
                    for (
                        let x = Math.floor(axis.min / day) * day;
                        x <= Math.ceil(axis.max / day) * day;
                        x += day
                    ) {
                        const last = plotBands.at(-1);
                        if (isWeekend(x) && !inWeekend) {
                            plotBands.push({
                                from: x,
                                color: {
                                    pattern: {
                                        path: 'M 0 10 L 10 0 M -1 1 L 1 -1 M 9 11 L 11 9',
                                        width: 10,
                                        height: 10,
                                        color: 'rgba(128,128,128,0.15)'
                                    }
                                }
                            });
                            inWeekend = true;
                        }
            
                        if (!isWeekend(x) && inWeekend && last) {
                            last.to = x;
                            inWeekend = false;
                        }
                    }
                    axis.options.plotBands = plotBands;
                }
            });
    """

    graph += f"""HighchartsGantt.ganttChart('{id_name}', options_{id_name});"""

    return f"""
            {html_figure}
            <script>
                {data_init}
                {graph}
            </script>
            """


