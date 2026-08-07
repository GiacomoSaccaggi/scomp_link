# -*- coding: utf-8 -*-
"""

██████╗ ███████╗██████╗  █████╗ ██████╗ ████████╗
██╔══██╗██╔════╝██╔══██╗██╔══██╗██╔══██╗╚══██╔══╝
██████╔╝█████╗  ██████╔╝██║  ██║██████╔╝   ██║
██╔══██╗██╔══╝  ██╔═══╝ ██║  ██║██╔══██╗   ██║
██║  ██║███████╗██║     ╚█████╔╝██║  ██║   ██║
╚═╝  ╚═╝╚══════╝╚═╝      ╚════╝ ╚═╝  ╚═╝   ╚═╝

██╗  ██╗████████╗███╗   ███╗██╗
██║  ██║╚══██╔══╝████╗ ████║██║
███████║   ██║   ██╔████╔██║██║
██╔══██║   ██║   ██║╚██╔╝██║██║
██║  ██║   ██║   ██║ ╚═╝ ██║███████╗
╚═╝  ╚═╝   ╚═╝   ╚═╝     ╚═╝╚══════╝
"""

import base64
import io
import json
import uuid
import warnings
from typing import Any

# Read constants from encrypted file
import os

import jwt
import pandas as pd
import plotly
from plotly.io.json import to_json_plotly

from scomp_link.utils.colors import MAIN, MAIN_DARK, MAIN_LIGHT
from scomp_link.utils.logger import get_logger

logger = get_logger(__name__)

_DEFAULT_FOOTER_CONTENT = """<footer><strong>About scomp-link</strong><br>
                        scomp-link is a general-purpose data science toolkit for preprocessing, model selection, and validation. It is dataset- and domain-agnostic.
                        Learn more in the project README and documentation.<br>
                        <strong>May the code be with you.</strong><br><br>
                        Copyright &copy; 2026 scomp-link contributors. All rights reserved.<br><br></footer>"""

_FOOTER_JS_BLOCK = """
                        <script>
                        // Quick and simple export target #table_id into a csv
                        function download_table_as_csv(table_id, separator = ',') {
                            var rows = document.querySelectorAll('table#' + table_id + ' tr');
                            var csv = [];
                            for (var i = 0; i < rows.length; i++) {
                                var row = [], cols = rows[i].querySelectorAll('td, th');
                                for (var j = 0; j < cols.length; j++) {
                                    var data = cols[j].innerText.replace(/(\\r\\n|\\n|\\r)/gm, '').replace(/(\\s\\s)/gm, ' ')
                                    data = data.replace(/"/g, '""');
                                    row.push('"' + data + '"');
                                }
                                csv.push(row.join(separator));
                            }
                            var csv_string = csv.join('\\n');
                            var filename = 'export_' + table_id + '_' + new Date().toLocaleDateString() + '.csv';
                            var link = document.createElement('a');
                            link.style.display = 'none';
                            link.setAttribute('target', '_blank');
                            link.setAttribute('href', 'data:text/csv;charset=utf-8,' + encodeURIComponent(csv_string));
                            link.setAttribute('download', filename);
                            document.body.appendChild(link);
                            link.click();
                            document.body.removeChild(link);
                        }

                        // Resize all Plotly charts within a container
                        function resizePlotsIn(container) {
                            if (!window.Plotly) return;
                            var plots = container.querySelectorAll('.js-plotly-plot');
                            plots.forEach(function(plot) {
                                Plotly.Plots.resize(plot);
                            });
                        }

                        // Collapsible sections with Plotly resize on open
                        var coll = document.getElementsByClassName("collapsiblemygs");
                        // Collapse all sections immediately (do not rely on DOMContentLoaded which may have already fired)
                        for (var ci = 0; ci < coll.length; ci++) {
                            var sibling = coll[ci].nextElementSibling;
                            if (sibling) sibling.style.display = 'none';
                        }
                        for (var i = 0; i < coll.length; i++) {
                          coll[i].addEventListener("click", function() {
                            this.classList.toggle("active");
                            var content = this.nextElementSibling;
                            if (content.style.display === "none") {
                              content.style.display = "block";
                              // Resize Plotly charts after section becomes visible
                              setTimeout(function() { resizePlotsIn(content); }, 50);
                              setTimeout(function() { resizePlotsIn(content); }, 300);
                            } else {
                              content.style.display = "none";
                            }
                          });
                        }

                        document.addEventListener("DOMContentLoaded", function() {
                            // Collapse all sections (fallback)
                            var contents = document.querySelectorAll('.content');
                            contents.forEach(function(content) {
                                content.style.display = 'none';
                            });

                            // Highcharts containers: ensure 100% width
                            document.querySelectorAll('.highcharts-container').forEach(function(item) {
                                item.style.width = '100%';
                            });

                            // Global ResizeObserver: auto-resize all Plotly charts on container resize
                            if (window.ResizeObserver && window.Plotly) {
                                var ro = new ResizeObserver(function(entries) {
                                    entries.forEach(function(entry) {
                                        var plot = entry.target.querySelector('.js-plotly-plot') || entry.target;
                                        if (plot && plot.classList.contains('js-plotly-plot')) {
                                            Plotly.Plots.resize(plot);
                                        }
                                    });
                                });
                                document.querySelectorAll('.plotly-graph-div').forEach(function(div) {
                                    ro.observe(div);
                                });
                                // Also observe for dynamically added charts
                                var bodyObserver = new MutationObserver(function(mutations) {
                                    mutations.forEach(function(m) {
                                        m.addedNodes.forEach(function(node) {
                                            if (node.nodeType === 1) {
                                                var divs = node.querySelectorAll ? node.querySelectorAll('.plotly-graph-div') : [];
                                                divs.forEach(function(d) { ro.observe(d); });
                                                if (node.classList && node.classList.contains('plotly-graph-div')) { ro.observe(node); }
                                            }
                                        });
                                    });
                                });
                                bodyObserver.observe(document.body, { childList: true, subtree: true });
                            }

                            // Fallback: resize all visible Plotly charts on window resize
                            window.addEventListener('resize', function() {
                                if (!window.Plotly) return;
                                document.querySelectorAll('.js-plotly-plot').forEach(function(plot) {
                                    Plotly.Plots.resize(plot);
                                });
                            });
                        });

                        </script>
                        </div><hr><br>"""


def _build_footer(footer_html=None):
    """Build the complete footer block (JS scripts + visual footer)."""
    return _FOOTER_JS_BLOCK + (footer_html or _DEFAULT_FOOTER_CONTENT)


class ScompLinkHTMLReport:
    def __init_subclass__(
        cls,
        title,
        font_family="Baloo 2",
        url_img_logo="",
        # optional logo URL (empty for neutral branding)
        # optional secondary logo URL (empty for neutral branding)
        url_background_header="https://giacomosaccaggi.github.io/deep-dives/sfondo.png",
        description="Automatic Report",
        author="scomp-link toolkit",
        language="en",
        main_color=MAIN,
        light_color=MAIN_LIGHT,
        dark_color=MAIN_DARK,
        footer_html=None,
    ):
        cls.dark_color = dark_color
        cls.light_color = light_color
        cls.main_color = main_color
        cls.html_title = f"<title>{title}</title>"
        cls.header = f"""<header>
                                    <h1>{title}</h1><br><br><br>
                                </header>"""
        cls.html_meta_info = f"""
                    <meta charset="utf-8">
                    <meta http-equiv="X-UA-Compatible" content="IE=edge">
                    <meta name="viewport" content="width=device-width, initial-scale=1">
                    <meta name="description" content="{description}">
                    <meta name="author" content="{author}">
                    """
        cls.html_layout = (
            """
            <link rel="shortcut icon" href="{url_img_logo}" />
            <script src="https://ajax.googleapis.com/ajax/libs/jquery/2.1.1/jquery.min.js"></script>
            <link href="https://cdnjs.cloudflare.com/ajax/libs/select2/4.0.6-rc.0/css/select2.min.css" rel="stylesheet"/>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/select2/4.0.6-rc.0/js/select2.min.js"></script>
            <script src="https://code.highcharts.com/gantt/highcharts-gantt.js"></script>
            <script>
                var HighchartsGantt = Highcharts; // Rinomina Highcharts Gantt per evitare conflitti
                // Ora puoi utilizzare Highcharts per Highcharts.js e HighchartsGantt per Highcharts Gantt
            </script>
            <script src="https://code.highcharts.com/highcharts.js"></script>
            <script src="https://code.highcharts.com/modules/streamgraph.js"></script>
            <script src="https://code.highcharts.com/modules/series-label.js"></script>
            <script src="https://code.highcharts.com/modules/annotations.js"></script>
            <script src="https://code.highcharts.com/modules/exporting.js"></script>
            <script src="https://code.highcharts.com/modules/export-data.js"></script>
            <script src="https://code.highcharts.com/modules/heatmap.js"></script>
            <script src="https://code.highcharts.com/gantt/modules/exporting.js"></script>
            <script src="https://code.highcharts.com/gantt/modules/pattern-fill.js"></script>
            <script src="https://code.highcharts.com/gantt/modules/accessibility.js"></script>
            <script>const colors = Highcharts.getOptions().colors;</script>

            <style>
            :root {
              --bg: #ffffff;
              --card: #f8fafc;
              --accent: {main_color};
              --accent2: {dark_color};
              --accent3: #0f9d58;
              --accent4: #e8590c;
              --accent5: #d6336c;
              --text: #1e293b;
              --dim: #64748b;
              --border: #e2e8f0;
              --radius: 8px;
              --baseFg: dimgray;
              --baseBg: white;
              --accentFg: {dark_color};
              --accentBg: {main_color};
            }
            *{margin:0;padding:0;box-sizing:border-box}
            html,body{font-family:{font_family}, system-ui, sans-serif;font-size:15px;line-height:1.6;color:var(--text)}
            html{overflow-x:hidden}
            h1{font-size:2.2rem;font-weight:800;background:linear-gradient(135deg,var(--accent),var(--accent2));-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent}
            h2{font-size:1.3rem;font-weight:700;color:var(--accent);margin:1rem 0 .5rem}
            h3{font-size:1rem;font-weight:600;color:var(--text)}
            h4{font-size:.9rem;color:var(--accent3)}
            h5{font-size:.8rem;color:var(--dim)}
            h6{font-size:.7rem;color:var(--dim)}
            p{margin:.5rem 0;text-align:justify}
            footer{text-align:center;padding:2rem 15%;color:var(--dim);font-size:.85rem}
            header{
              position:relative;
              background-position:center;
              background-repeat:no-repeat;
              background-size:cover;
              background-image:url({url_background_header});
              height:50%;
              width:100%;
              color:white;
              text-align:left;
              padding-top:5%;
              padding-left:15%;
            }
            header h1{-webkit-text-fill-color:white;background:none}
            .user-select-none .svg-container{height:100%;width:100%}
            .report{
              padding:2rem 10%;
              max-width:1400px;
              margin:0 auto;
              background:var(--bg);
            }
            input[type=button],input[type=submit],input[type=reset]{
              padding:12px 24px;
              text-decoration:none;
              margin:4px 2px;
              cursor:pointer;
              display:inline-block;
              border-radius:var(--radius);
              background-color:white;
              color:var(--accent2);
              border:2px solid var(--accent);
              text-align:center;
              transition:all .3s ease;
              font-weight:600;
              font-size:.85rem;
            }
            input[type=button]:hover,input[type=submit]:hover,input[type=reset]:hover{
              background-color:var(--accent);
              color:white;
              box-shadow:0 4px 12px rgba(0,0,0,.1);
            }
            .plotly-graph-div{width:100%;min-height:200px;overflow:hidden}
            #container{height:600px}
            .highcharts-label-icon{opacity:0.5}
            .highcharts-figure,.highcharts-data-table table{min-width:310px;max-width:100%;overflow:auto;margin:1em auto}
            .highcharts-data-table table{font-family:inherit;border-collapse:collapse;border:1px solid var(--border);margin:10px auto;text-align:center;width:100%}
            .highcharts-figure{width:100%;height:auto}
            .highcharts-data-table caption{padding:1em 0;font-size:1.1em;color:var(--dim)}
            .highcharts-data-table th{font-weight:600;padding:0.5em}
            .highcharts-data-table td,.highcharts-data-table th,.highcharts-data-table caption{padding:0.5em}
            .highcharts-data-table thead tr,.highcharts-data-table tr:nth-child(even){background:var(--card)}
            .highcharts-data-table tr:hover{background:rgba(0,94,184,.04)}
            #table-wrapper{position:relative}
            #table-scroll{max-height:500px;overflow:auto;margin-top:20px;border:1px solid var(--border);border-radius:var(--radius)}
            #table-wrapper table{width:100%;border-collapse:collapse}
            #table-wrapper table *{color:var(--text)}
            #table-wrapper table thead th{background:var(--accent);color:white;text-transform:uppercase;font-size:.7rem;letter-spacing:.04em;padding:.6rem .8rem}
            #table-wrapper table td{padding:.45rem .8rem;border-bottom:1px solid var(--border);font-size:.85rem}
            #table-wrapper table tr:nth-child(even){background:var(--card)}
            #table-wrapper table tr:hover{background:rgba(0,94,184,.04)}
            .column_result_save{float:left;width:50%;padding:10px}
            code{color:var(--accent2);background:rgba(110,55,250,.07);padding:1px 5px;border-radius:3px;font-size:.85em}
            select{
              font:400 12px/1.3 sans-serif;
              -webkit-appearance:none;
              appearance:none;
              color:var(--baseFg);
              border:1px solid var(--border);
              line-height:1;
              outline:0;
              padding:0.65em 2.5em 0.55em 0.75em;
              border-radius:var(--radius);
              background-color:var(--baseBg);
              background-image:linear-gradient(var(--baseFg),var(--baseFg)),linear-gradient(-135deg,transparent 50%,var(--accentBg) 50%),linear-gradient(-225deg,transparent 50%,var(--accentBg) 50%),linear-gradient(var(--accentBg) 42%,var(--accentFg) 42%);
              background-repeat:no-repeat;
              background-size:1px 100%,20px 22px,20px 22px,20px 100%;
              background-position:right 20px center,right bottom,right bottom,right bottom;
            }
            select:hover{
              background-image:linear-gradient(var(--accentFg),var(--accentFg)),linear-gradient(-135deg,transparent 50%,var(--accentFg) 50%),linear-gradient(-225deg,transparent 50%,var(--accentFg) 50%),linear-gradient(var(--accentFg) 42%,var(--accentBg) 42%);
            }
            .collapsiblemygs{
              background-color:var(--card);
              color:var(--text);
              cursor:pointer;
              padding:14px 20px;
              width:100%;
              border:1px solid var(--border);
              border-radius:var(--radius);
              text-align:left;
              outline:none;
              font-size:.95rem;
              font-weight:600;
              margin-top:.75rem;
              transition:all .2s ease;
            }
            .collapsiblemygs:hover,.active{
              background-color:rgba(0,94,184,.04);
              border-color:var(--accent);
              color:var(--accent);
            }
            .content{
              padding:1.2rem 1.5rem;
              display:block;
              overflow:hidden;
              background-color:var(--card);
              border:1px solid var(--border);
              border-top:none;
              border-radius:0 0 var(--radius) var(--radius);
            }
            @media only screen and (max-width:1024px){
              .report{padding:1.5rem 5%}
            }
            @media only screen and (max-width:700px){
              h2{font-size:1.1rem}
              .collapsiblemygs{font-size:.85rem;padding:12px 16px}
            }
            @media print{
              .collapsiblemygs{
                page-break-before:always;
                background-color:transparent !important;
                color:black !important;
                font-size:24px !important;
                font-weight:bold;
                border-bottom:2px solid var(--accentBg);
                border:none !important;
                padding:0 0 10px 0 !important;
                margin-top:30px !important;
              }
              .content{
                display:block !important;
                background-color:white !important;
                border:none !important;
                padding:0 !important;
              }
              select,input[type="submit"],input[type="button"],label{display:none !important}
              .print-grid-container{display:flex !important;flex-wrap:wrap !important;justify-content:space-between !important}
              .print-grid-item{display:block !important;width:48% !important;page-break-inside:avoid}
              body{-webkit-print-color-adjust:exact;print-color-adjust:exact}
            }
            </style>
            <script src="https://cdn.plot.ly/plotly-2.9.0.min.js"></script>
        """.replace("{font_family}", font_family)
            .replace("{url_img_logo}", url_img_logo)
            .replace("{url_background_header}", url_background_header)
            .replace("{main_color}", cls.main_color)
            .replace("{dark_color}", cls.dark_color)
            .replace("{light_color}", cls.light_color)
        )
        cls.footer = _build_footer(footer_html)
        cls.html_report = ""
        cls.section_just_open = False
        cls.lan = language

    def __init__(
        self,
        title,
        font_family="Baloo 2",
        url_img_logo="",
        # optional logo URL (empty for neutral branding)
        # optional secondary logo URL (empty for neutral branding)
        url_background_header="https://giacomosaccaggi.github.io/deep-dives/sfondo.png",
        description="Automatic Report",
        author="scomp-link toolkit",
        language="en",
        main_color=MAIN,
        light_color=MAIN_LIGHT,
        dark_color=MAIN_DARK,
        footer_html=None,
    ):
        self.dark_color = dark_color
        self.light_color = light_color
        self.main_color = main_color
        self.html_title = f"<title>{title}</title>"
        self.header = f"""<header>
                            <h1>{title}</h1><br><br><br>
                        </header>"""
        self.html_meta_info = f"""
            <meta charset="utf-8">
            <meta http-equiv="X-UA-Compatible" content="IE=edge">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <meta name="description" content="{description}">
            <meta name="author" content="{author}">
            """
        self.html_layout = (
            """
            <link rel="shortcut icon" href="{url_img_logo}" />
            <script src="https://ajax.googleapis.com/ajax/libs/jquery/2.1.1/jquery.min.js"></script>
            <link href="https://cdnjs.cloudflare.com/ajax/libs/select2/4.0.6-rc.0/css/select2.min.css" rel="stylesheet"/>
            <script src="https://cdnjs.cloudflare.com/ajax/libs/select2/4.0.6-rc.0/js/select2.min.js"></script>
            <script src="https://code.highcharts.com/gantt/highcharts-gantt.js"></script>
            <script>
                var HighchartsGantt = Highcharts; // Rinomina Highcharts Gantt per evitare conflitti
                // Ora puoi utilizzare Highcharts per Highcharts.js e HighchartsGantt per Highcharts Gantt
            </script>
            <script src="https://code.highcharts.com/highcharts.js"></script>
            <script src="https://code.highcharts.com/modules/streamgraph.js"></script>
            <script src="https://code.highcharts.com/modules/series-label.js"></script>
            <script src="https://code.highcharts.com/modules/annotations.js"></script>
            <script src="https://code.highcharts.com/modules/exporting.js"></script>
            <script src="https://code.highcharts.com/modules/export-data.js"></script>
            <script src="https://code.highcharts.com/modules/heatmap.js"></script>
            <script src="https://code.highcharts.com/gantt/modules/exporting.js"></script>
            <script src="https://code.highcharts.com/gantt/modules/pattern-fill.js"></script>
            <script src="https://code.highcharts.com/gantt/modules/accessibility.js"></script>
            <script>const colors = Highcharts.getOptions().colors;</script>

            <style>
            :root {
              --bg: #ffffff;
              --card: #f8fafc;
              --accent: {main_color};
              --accent2: {dark_color};
              --accent3: #0f9d58;
              --accent4: #e8590c;
              --accent5: #d6336c;
              --text: #1e293b;
              --dim: #64748b;
              --border: #e2e8f0;
              --radius: 8px;
              --baseFg: dimgray;
              --baseBg: white;
              --accentFg: {dark_color};
              --accentBg: {main_color};
            }
            *{margin:0;padding:0;box-sizing:border-box}
            html,body{font-family:{font_family}, system-ui, sans-serif;font-size:15px;line-height:1.6;color:var(--text)}
            html{overflow-x:hidden}
            h1{font-size:2.2rem;font-weight:800;background:linear-gradient(135deg,var(--accent),var(--accent2));-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent}
            h2{font-size:1.3rem;font-weight:700;color:var(--accent);margin:1rem 0 .5rem}
            h3{font-size:1rem;font-weight:600;color:var(--text)}
            h4{font-size:.9rem;color:var(--accent3)}
            h5{font-size:.8rem;color:var(--dim)}
            h6{font-size:.7rem;color:var(--dim)}
            p{margin:.5rem 0;text-align:justify}
            footer{text-align:center;padding:2rem 15%;color:var(--dim);font-size:.85rem}
            header{
              position:relative;
              background-position:center;
              background-repeat:no-repeat;
              background-size:cover;
              background-image:url({url_background_header});
              height:50%;
              width:100%;
              color:white;
              text-align:left;
              padding-top:5%;
              padding-left:15%;
            }
            header h1{-webkit-text-fill-color:white;background:none}
            .user-select-none .svg-container{height:100%;width:100%}
            .report{
              padding:2rem 10%;
              max-width:1400px;
              margin:0 auto;
              background:var(--bg);
            }
            input[type=button],input[type=submit],input[type=reset]{
              padding:12px 24px;
              text-decoration:none;
              margin:4px 2px;
              cursor:pointer;
              display:inline-block;
              border-radius:var(--radius);
              background-color:white;
              color:var(--accent2);
              border:2px solid var(--accent);
              text-align:center;
              transition:all .3s ease;
              font-weight:600;
              font-size:.85rem;
            }
            input[type=button]:hover,input[type=submit]:hover,input[type=reset]:hover{
              background-color:var(--accent);
              color:white;
              box-shadow:0 4px 12px rgba(0,0,0,.1);
            }
            .plotly-graph-div{width:100%;min-height:200px;overflow:hidden}
            #container{height:600px}
            .highcharts-label-icon{opacity:0.5}
            .highcharts-figure,.highcharts-data-table table{min-width:310px;max-width:100%;overflow:auto;margin:1em auto}
            .highcharts-data-table table{font-family:inherit;border-collapse:collapse;border:1px solid var(--border);margin:10px auto;text-align:center;width:100%}
            .highcharts-figure{width:100%;height:auto}
            .highcharts-data-table caption{padding:1em 0;font-size:1.1em;color:var(--dim)}
            .highcharts-data-table th{font-weight:600;padding:0.5em}
            .highcharts-data-table td,.highcharts-data-table th,.highcharts-data-table caption{padding:0.5em}
            .highcharts-data-table thead tr,.highcharts-data-table tr:nth-child(even){background:var(--card)}
            .highcharts-data-table tr:hover{background:rgba(0,94,184,.04)}
            #table-wrapper{position:relative}
            #table-scroll{max-height:500px;overflow:auto;margin-top:20px;border:1px solid var(--border);border-radius:var(--radius)}
            #table-wrapper table{width:100%;border-collapse:collapse}
            #table-wrapper table *{color:var(--text)}
            #table-wrapper table thead th{background:var(--accent);color:white;text-transform:uppercase;font-size:.7rem;letter-spacing:.04em;padding:.6rem .8rem}
            #table-wrapper table td{padding:.45rem .8rem;border-bottom:1px solid var(--border);font-size:.85rem}
            #table-wrapper table tr:nth-child(even){background:var(--card)}
            #table-wrapper table tr:hover{background:rgba(0,94,184,.04)}
            .column_result_save{float:left;width:50%;padding:10px}
            code{color:var(--accent2);background:rgba(110,55,250,.07);padding:1px 5px;border-radius:3px;font-size:.85em}
            select{
              font:400 12px/1.3 sans-serif;
              -webkit-appearance:none;
              appearance:none;
              color:var(--baseFg);
              border:1px solid var(--border);
              line-height:1;
              outline:0;
              padding:0.65em 2.5em 0.55em 0.75em;
              border-radius:var(--radius);
              background-color:var(--baseBg);
              background-image:linear-gradient(var(--baseFg),var(--baseFg)),linear-gradient(-135deg,transparent 50%,var(--accentBg) 50%),linear-gradient(-225deg,transparent 50%,var(--accentBg) 50%),linear-gradient(var(--accentBg) 42%,var(--accentFg) 42%);
              background-repeat:no-repeat;
              background-size:1px 100%,20px 22px,20px 22px,20px 100%;
              background-position:right 20px center,right bottom,right bottom,right bottom;
            }
            select:hover{
              background-image:linear-gradient(var(--accentFg),var(--accentFg)),linear-gradient(-135deg,transparent 50%,var(--accentFg) 50%),linear-gradient(-225deg,transparent 50%,var(--accentFg) 50%),linear-gradient(var(--accentFg) 42%,var(--accentBg) 42%);
            }
            .collapsiblemygs{
              background-color:var(--card);
              color:var(--text);
              cursor:pointer;
              padding:14px 20px;
              width:100%;
              border:1px solid var(--border);
              border-radius:var(--radius);
              text-align:left;
              outline:none;
              font-size:.95rem;
              font-weight:600;
              margin-top:.75rem;
              transition:all .2s ease;
            }
            .collapsiblemygs:hover,.active{
              background-color:rgba(0,94,184,.04);
              border-color:var(--accent);
              color:var(--accent);
            }
            .content{
              padding:1.2rem 1.5rem;
              display:block;
              overflow:hidden;
              background-color:var(--card);
              border:1px solid var(--border);
              border-top:none;
              border-radius:0 0 var(--radius) var(--radius);
            }
            @media only screen and (max-width:1024px){
              .report{padding:1.5rem 5%}
            }
            @media only screen and (max-width:700px){
              h2{font-size:1.1rem}
              .collapsiblemygs{font-size:.85rem;padding:12px 16px}
            }
            @media print{
              .collapsiblemygs{
                page-break-before:always;
                background-color:transparent !important;
                color:black !important;
                font-size:24px !important;
                font-weight:bold;
                border-bottom:2px solid var(--accentBg);
                border:none !important;
                padding:0 0 10px 0 !important;
                margin-top:30px !important;
              }
              .content{
                display:block !important;
                background-color:white !important;
                border:none !important;
                padding:0 !important;
              }
              select,input[type="submit"],input[type="button"],label{display:none !important}
              .print-grid-container{display:flex !important;flex-wrap:wrap !important;justify-content:space-between !important}
              .print-grid-item{display:block !important;width:48% !important;page-break-inside:avoid}
              body{-webkit-print-color-adjust:exact;print-color-adjust:exact}
            }
            </style>
            <script src="https://cdn.plot.ly/plotly-2.9.0.min.js"></script>
        """.replace("{font_family}", font_family)
            .replace("{url_img_logo}", url_img_logo)
            .replace("{url_background_header}", url_background_header)
            .replace("{main_color}", self.main_color)
            .replace("{dark_color}", self.dark_color)
            .replace("{light_color}", self.light_color)
        )
        self.footer = _build_footer(footer_html)
        self.html_report = ""
        self.section_just_open = False
        self.lan = language

    def single_plotly(self, fig: "plotly.graph_objs._figure.Figure", title: str, plotdivid: str | None = None) -> str:
        """
        This function it is usefull to create a single plot in html.
        Returns HTML plotly code.
        :type plotdivid: object
        """
        fig_json = to_json_plotly(fig)
        assert fig_json is not None
        fig_dict = json.loads(fig_json)
        jdata = to_json_plotly(fig_dict.get("data", []))
        layout = fig_dict.get("layout", {})
        layout["autosize"] = True
        layout.pop("width", None)  # remove fixed width if set by user
        jlayout = to_json_plotly(layout)
        jconfig = to_json_plotly({"responsive": True})
        if plotdivid is None:
            plotdivid = title.replace(" ", "_")
            for p in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~":
                plotdivid = plotdivid.replace(p, "_")

        options = ""  # autosize handled by responsive config + ResizeObserver
        script = """\
                <h2>{title}</h2>
                <div id="{id}" class="plotly-graph-div"></div>
                <script>
                        Plotly.newPlot(\n
                            "{id}",\n
                            {data},\n
                            {layout},\n
                            {config}
                        )
                        {options}
                </script>
                    """.format(title=title, id=plotdivid, data=jdata, layout=jlayout, config=jconfig, options=options)

        return script

    def select_plotly(self, figures_dict: dict, title: str, labels="Choose a label") -> str:
        """
        .. deprecated::
            Use :meth:`add_cascading_content` instead.

        Returns HTML plotly code with select dropdown.

        :param figures_dict: dict mapping label(s) to Plotly figures
        :param title: str - section title
        :param labels: str or list[str] - dropdown label(s)
        :return: empty string (content is appended directly to the report)
        """
        warnings.warn(
            "select_plotly() is deprecated, use add_cascading_content() instead",
            DeprecationWarning,
            stacklevel=2,
        )

        # Convert old format to new format
        keys = list(figures_dict.keys())
        multiple = isinstance(keys[0], tuple)

        if multiple:
            n_filters = len(keys[0])
            if isinstance(labels, (list, tuple)) and len(labels) == n_filters:
                dim_labels = list(labels)
            else:
                dim_labels = [f"Choose a label"] * n_filters

            # Gather unique options per dimension
            options_per_dim: list[list[str]] = [[] for _ in range(n_filters)]
            for key_tuple in keys:
                for i, k in enumerate(key_tuple):
                    if k not in options_per_dim[i]:
                        options_per_dim[i].append(k)

            dimensions = [
                {"label": dim_labels[i], "options": options_per_dim[i]} for i in range(n_filters)
            ]
            content_map = {k: v for k, v in figures_dict.items()}
        else:
            dim_label = labels if isinstance(labels, str) else labels[0]
            dimensions = [{"label": dim_label, "options": [str(k) for k in keys]}]
            content_map = {(str(k),): v for k, v in figures_dict.items()}

        self.add_cascading_content(title, dimensions, content_map)
        return ""

    def add_graph_to_report(self, fig: "plotly.graph_objs._figure.Figure", title: str):
        """
        Add graph to report
        :param fig: plotly.graph_objs._figure.Figure
        :param title: str
        :return:

        ## example
        demo_report = NielsenHTMLreport('My fisrt REPORT') # if you don't have just created
        import plotly.express as px
        fig = px.scatter(x=range(10), y=range(10))
        demo_report.add_graph_to_report(fig, 'My first Graph')
        """
        self.html_report += self.single_plotly(fig, title)
        logger.info("Added graph to report!")

    def add_matplotlib_graph_to_report(self, fig, title: str, dpi: int = 150, img_format: str = "png"):
        """
        Add a matplotlib figure to the report as a base64-encoded image.

        :param fig: matplotlib.figure.Figure - the matplotlib figure
        :param title: str - title displayed above the image
        :param dpi: int - resolution of the exported image (default 150)
        :param img_format: str - image format, 'png' or 'svg' (default 'png')

        ## example
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(range(10), range(10))
        report.add_matplotlib_graph_to_report(fig, 'My Matplotlib Graph')
        """
        buf = io.BytesIO()
        fig.savefig(buf, format=img_format, dpi=dpi, bbox_inches="tight")
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode("utf-8")
        buf.close()
        mime = "image/svg+xml" if img_format == "svg" else f"image/{img_format}"
        self.html_report += f"<h2>{title}</h2>"
        self.html_report += (
            f'<img src="data:{mime};base64,{img_base64}" style="width:100%;max-width:100%;" alt="{title}">'
        )
        logger.info("Added matplotlib graph to report!")

    def add_image_to_report(self, image_path: str, title: str):
        """
        Add a local image file to the report as a base64-encoded image.

        :param image_path: str - absolute or relative path to the image file
        :param title: str - title displayed above the image

        ## example
        report.add_image_to_report('/path/to/image.png', 'My Image')
        """
        ext = os.path.splitext(image_path)[1].lower().lstrip(".")
        mime = "image/svg+xml" if ext == "svg" else f"image/{ext}"
        with open(image_path, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode("utf-8")
        self.html_report += f"<h2>{title}</h2>"
        self.html_report += (
            f'<img src="data:{mime};base64,{img_base64}" style="width:100%;max-width:100%;" alt="{title}">'
        )
        logger.info("Added image to report!")

    def add_rawgraphs_to_report(self, svg_string: str, title: str):
        """
        Add a RAWGraphs-style SVG chart to the report.

        :param svg_string: str - SVG markup string returned by rawgraphs functions
        :param title: str - title displayed above the chart

        ## example
        from scomp_link.utils.rawgraphs import barchart
        svg = barchart(['A', 'B', 'C'], [10, 20, 30], 'My Chart')
        report.add_rawgraphs_to_report(svg, 'Bar Chart Example')
        """
        self.html_report += f"<h2>{title}</h2>"
        self.html_report += f'<div style="width:100%;overflow-x:auto;">{svg_string}</div>'
        logger.info("Added RAWGraphs SVG to report!")

    def add_graph(self, fig: Any, title: str = "", **kwargs: Any) -> None:
        """
        Add a chart to the report. Automatically detects the chart type.

        Supported types (detected automatically):
        - plotly.graph_objs.Figure       → embedded as interactive Plotly chart
        - matplotlib.figure.Figure       → embedded as base64 image
        - matplotlib.axes.Axes           → extracts .figure, then embedded as base64 image
          (seaborn returns Axes — just pass the result of sns.barplot() etc. directly)
        - str starting with "<svg"       → RAWGraphs SVG (from scomp_link.utils.rawgraphs)
        - str containing "Highcharts"    → Highcharts HTML snippet (from scomp_link.utils.highcharts)
        - str (other)                    → raw HTML injected verbatim

        :param fig: the chart object or HTML string
        :param title: str - optional title displayed above the chart
        :param kwargs: extra args forwarded to the underlying method
                       (e.g. dpi=150, img_format="png" for matplotlib)

        ## example
        import plotly.express as px
        import matplotlib.pyplot as plt
        import seaborn as sns
        from scomp_link.utils.rawgraphs import treemap
        from scomp_link.utils.highcharts import streamgraphs

        report.add_graph(px.scatter(df, x="x", y="y"), "Scatter")
        report.add_graph(plt.figure(), "Matplotlib")
        report.add_graph(sns.barplot(data=df, x="x", y="y"), "Seaborn")
        report.add_graph(treemap(data, "x", "value"), "Treemap")
        report.add_graph(streamgraphs("Stream", dates, series), "Stream")
        """
        # --- Plotly ---
        try:
            import plotly.graph_objs as _go
            if isinstance(fig, _go.Figure):
                self.add_graph_to_report(fig, title)
                return
        except ImportError:
            pass

        # --- Matplotlib Figure ---
        try:
            import matplotlib.figure as _mfig
            if isinstance(fig, _mfig.Figure):
                self.add_matplotlib_graph_to_report(fig, title, **kwargs)
                return
        except ImportError:
            pass

        # --- Matplotlib Axes (seaborn returns Axes) ---
        try:
            import matplotlib.axes as _maxes
            if isinstance(fig, _maxes.Axes):
                self.add_matplotlib_graph_to_report(fig.figure, title, **kwargs)
                return
        except ImportError:
            pass

        # --- String-based: SVG, Highcharts, raw HTML ---
        if isinstance(fig, str):
            stripped = fig.lstrip()
            if stripped.startswith("<svg") or stripped.startswith("<?xml"):
                # RAWGraphs SVG
                self.add_rawgraphs_to_report(fig, title)
                return
            if "Highcharts" in fig or "HighchartsGantt" in fig:
                # Highcharts HTML snippet
                if title:
                    self.html_report += f"<h2>{title}</h2>"
                self.add_highcharts(fig)
                return
            # Generic HTML
            if title:
                self.html_report += f"<h2>{title}</h2>"
            self.add_html(fig)
            return

        raise TypeError(
            f"add_graph() does not support type {type(fig).__name__}. "
            "Supported: plotly Figure, matplotlib Figure/Axes, str (SVG / Highcharts HTML / raw HTML)."
        )

    def add_many_plots_with_selection_box_to_report(self, figures_dict: dict, title: str, **kwargs):
        """
        .. deprecated::
            Use :meth:`add_cascading_content` instead.

        Add multiple Plotly graphs to the report with a dropdown selector.

        :param figures_dict: dict mapping label(s) to Plotly figures
        :param title: str - section title
        :param kwargs: optional 'labels' parameter for dropdown label(s)

        ## example
        import plotly.express as px
        fig1 = px.scatter(x=range(10), y=range(10))
        fig2 = px.scatter(x=range(20), y=range(20))
        figures_dict = {'First': fig1, 'Second': fig2}
        report.add_many_plots_with_selection_box_to_report(figures_dict, 'My Graphs')
        """
        warnings.warn(
            "add_many_plots_with_selection_box_to_report() is deprecated, use add_cascading_content() instead",
            DeprecationWarning,
            stacklevel=2,
        )
        labels = kwargs.get("labels", "Choose a label")
        self.select_plotly(figures_dict, title, labels=labels)
        logger.info("Added graph to report!")

    def open_section(self, section_title: str, ingore_multi_section=False) -> None:
        if not self.section_just_open or ingore_multi_section:
            self.html_report += f'<button class="collapsiblemygs">{section_title}</button> <div class="content">'
            self.section_just_open = True
            logger.info("Open section to report!")
        else:
            logger.info("Warning you already have an open section")

    def close_section(self, ingore_multi_section=False) -> None:
        if self.section_just_open or ingore_multi_section:
            self.html_report += "</div>"
            self.section_just_open = False
            logger.info("Close section to report!")
        else:
            logger.info("Warning you did not open section yet")

    def add_title(self, title: str) -> None:
        self.html_report += f"<h2>{title}</h2>"
        logger.info("Added title to report!")

    def add_subtitle(self, subtitle: str) -> None:
        """
        Add an <h3> subtitle to the report.

        :param subtitle: str - subtitle text

        ## example
        report.add_subtitle("Model Details")
        """
        self.html_report += f"<h3>{subtitle}</h3>"
        logger.info("Added subtitle to report!")

    def add_text(self, text: str) -> None:
        self.html_report += f"<p>{text}</p>"
        logger.info("Added text to report!")

    def add_highcharts(self, html_snippet: str) -> None:
        """
        Add a Highcharts chart to the report.

        Use this for HTML snippets returned by highcharts module functions:
        streamgraphs(), calendar_heatmap(), calendar_gantt(), area_chart(), etc.

        :param html_snippet: str - HTML string returned by a scomp_link.utils.highcharts function

        ## example
        from scomp_link.utils.highcharts import streamgraphs
        html = streamgraphs("My Chart", dates, series_dict)
        report.add_highcharts(html)
        """
        self.html_report += html_snippet
        logger.info("Added Highcharts chart to report!")

    def add_html(self, html: str) -> None:
        """
        Inject arbitrary HTML directly into the report body.

        Use this only when no other method covers your use case.
        Prefer the specific methods instead:
        - Text content   → add_title(), add_subtitle(), add_text()
        - Plotly charts  → add_graph_to_report()
        - Highcharts     → add_highcharts()
        - RAWGraphs SVG  → add_rawgraphs_to_report()
        - DataFrames     → add_dataframe()
        - Images         → add_image_to_report()

        What this affects: the HTML is appended verbatim inside the <body>
        of the report. Malformed HTML here can break the entire report layout.

        :param html: str - raw HTML string to inject

        ## example
        report.add_html('<div class="custom-box">Custom content</div>')
        """
        self.html_report += html
        logger.info("Added raw HTML to report!")

    def add_cascading_content(
        self, title: str, dimensions: list[dict], content_map: dict, cascade: bool = False
    ) -> None:
        """
        Add interactive content with cascading dropdown selectors.

        Each combination of dropdown selections maps to a content block (HTML string
        or Plotly Figure). Only one block is visible at a time.

        :param title: str - heading displayed above the dropdowns
        :param dimensions: list[dict] - each dict has keys "label" (str) and "options" (list[str])
        :param content_map: dict - keys are tuples of option strings, values are HTML strings or Plotly Figures
        :param cascade: bool - if True, child dropdown options are filtered based on parent selection

        ## example
        import plotly.express as px
        fig1 = px.scatter(x=[1,2,3], y=[1,2,3])
        fig2 = px.scatter(x=[1,2,3], y=[3,2,1])
        report.add_cascading_content(
            "My Charts",
            [{"label": "Category", "options": ["A", "B"]}],
            {("A",): fig1, ("B",): fig2},
        )
        """
        import plotly.graph_objects as go
        import plotly.io as pio

        uid = uuid.uuid4().hex[:8]

        def _sanitize(s: str) -> str:
            return s.replace("-", "_").replace(".", "_").replace(" ", "_")

        # Build content divs
        content_divs = ""
        first = True
        for key_tuple, content in content_map.items():
            sanitized_key = "___".join(_sanitize(str(k)) for k in key_tuple)
            div_id = f"wrap_{uid}_{sanitized_key}"
            display = "block" if first else "none"
            first = False

            if isinstance(content, go.Figure):
                inner_html = pio.to_html(
                    content, include_plotlyjs=False, full_html=False, config={"responsive": True}
                )
            else:
                inner_html = str(content)

            content_divs += f'<div id="{div_id}" style="display:{display};overflow:visible;">{inner_html}</div>\n'

        # Build select elements
        selects_html = ""
        for i, dim in enumerate(dimensions):
            sel_id = f"sel_{i}_{uid}"
            options_html = "".join(
                f'<option value="{_sanitize(opt)}">{opt}</option>' for opt in dim["options"]
            )
            selects_html += (
                f'<label for="{sel_id}" style="margin-right:6px;font-weight:600;">{dim["label"]}:</label>'
                f'<select id="{sel_id}" onchange="update_{uid}()" '
                f'style="margin-right:12px;padding:4px 8px;border-radius:4px;border:1px solid #ccc;">'
                f"{options_html}</select>\n"
            )

        # Build cascade JS (filter child options based on parent selections)
        cascade_js = ""
        if cascade and len(dimensions) > 1:
            # Build mapping: parent_value -> available child options for each child dimension
            cascade_map: dict[int, dict[str, list[str]]] = {}
            for child_idx in range(1, len(dimensions)):
                parent_to_children: dict[str, set[str]] = {}
                for key_tuple in content_map.keys():
                    parent_key = "___".join(_sanitize(str(k)) for k in key_tuple[:child_idx])
                    child_val = str(key_tuple[child_idx])
                    if parent_key not in parent_to_children:
                        parent_to_children[parent_key] = set()
                    parent_to_children[parent_key].add(child_val)
                cascade_map[child_idx] = {k: sorted(v) for k, v in parent_to_children.items()}

            cascade_js = f"var cascade_{uid} = {json.dumps({str(k): v for k, v in cascade_map.items()})};\n"
            cascade_js += f"""
            function update_cascade_{uid}() {{
                var cas = cascade_{uid};
                for (var ci = 1; ci < {len(dimensions)}; ci++) {{
                    var parentKey = '';
                    for (var pi = 0; pi < ci; pi++) {{
                        if (pi > 0) parentKey += '___';
                        parentKey += document.getElementById('sel_' + pi + '_{uid}').value;
                    }}
                    var sel = document.getElementById('sel_' + ci + '_{uid}');
                    var opts = (cas[String(ci)] && cas[String(ci)][parentKey]) || [];
                    var curVal = sel.value;
                    sel.innerHTML = '';
                    for (var oi = 0; oi < opts.length; oi++) {{
                        var o = document.createElement('option');
                        o.value = opts[oi].replace(/-/g,'_').replace(/\\./g,'_').replace(/ /g,'_');
                        o.textContent = opts[oi];
                        sel.appendChild(o);
                    }}
                    if (opts.map(function(x){{return x.replace(/-/g,'_').replace(/\\./g,'_').replace(/ /g,'_');}}).indexOf(curVal) >= 0) {{
                        sel.value = curVal;
                    }}
                }}
            }}
            """

        # Build update JS
        key_parts = " + '___' + ".join(
            f"document.getElementById('sel_{i}_{uid}').value" for i in range(len(dimensions))
        )

        script = f"""
<script>
{cascade_js}
function update_{uid}() {{
    {"update_cascade_" + uid + "();" if cascade and len(dimensions) > 1 else ""}
    var k = {key_parts};
    document.querySelectorAll('[id^="wrap_{uid}_"]').forEach(function(d) {{ d.style.display = 'none'; }});
    var tk = k.replace(/-/g, '_').replace(/\\./g, '_').replace(/ /g, '_');
    var tgt = document.getElementById('wrap_{uid}_' + tk);
    if (tgt) {{
        tgt.style.display = 'block';
        tgt.style.overflow = 'visible';
        setTimeout(function() {{ window.dispatchEvent(new Event('resize')); }}, 150);
        setTimeout(function() {{ window.dispatchEvent(new Event('resize')); }}, 500);
    }}
}}
document.addEventListener('DOMContentLoaded', function() {{ update_{uid}(); }});
setTimeout(function() {{ update_{uid}(); }}, 500);
</script>
"""

        self.html_report += f"<h2>{title}</h2>\n"
        self.html_report += f'<div style="margin-bottom:12px;">{selects_html}</div>\n'
        self.html_report += content_divs
        self.html_report += script
        logger.info("Added cascading content to report!")

    def add_dataframe(self, df, title: str, limit_max=2000, thresholds=None) -> None:
        """
        Add a DataFrame as an HTML table to the report.

        :param df: pandas or polars DataFrame
        :param title: str - table title and CSV download ID
        :param limit_max: int - maximum rows before skipping rendering (default 2000)
        :param thresholds: dict[str, tuple[float, float, bool]] or None - color-coding thresholds per column.
            Format: {"col": (good_threshold, bad_threshold, higher_is_better)}.
            If higher_is_better=True: value > good → green, bad < value ≤ good → orange, value ≤ bad → red.
            If higher_is_better=False: value < good → green, good ≤ value < bad → orange, value ≥ bad → red.
        """
        # Accept polars DataFrames
        if hasattr(df, "to_pandas"):
            df = df.to_pandas()

        if len(df) < limit_max:
            table_id = title.replace(" ", "")
            self.html_report += (
                '<a href="#" onclick="download_table_as_csv('
                + f"'{table_id}'"
                + ');">Download as CSV</a>'
            )

            if thresholds is not None:
                # Build custom HTML table with threshold coloring
                tab = f'<table id="{table_id}" class="scomp-table">\n<thead><tr>'
                for col in df.columns:
                    tab += f"<th>{col}</th>"
                tab += "</tr></thead>\n<tbody>\n"
                for _, row in df.iterrows():
                    tab += "<tr>"
                    for col in df.columns:
                        val = row[col]
                        bg = ""
                        if col in thresholds:
                            good, bad, higher_is_better = thresholds[col]
                            if pd.isna(val):
                                bg = "rgba(100,100,100,0.2)"
                            elif higher_is_better:
                                if val > good:
                                    bg = "rgba(52,211,153,0.3)"
                                elif val > bad:
                                    bg = "rgba(251,146,60,0.3)"
                                else:
                                    bg = "rgba(239,68,68,0.3)"
                            else:
                                if val < good:
                                    bg = "rgba(52,211,153,0.3)"
                                elif val < bad:
                                    bg = "rgba(251,146,60,0.3)"
                                else:
                                    bg = "rgba(239,68,68,0.3)"
                        style_attr = f' style="background-color:{bg};"' if bg else ""
                        display_val = "" if pd.isna(val) else val
                        tab += f"<td{style_attr}>{display_val}</td>"
                    tab += "</tr>\n"
                tab += "</tbody></table>"
            else:
                tab = df.to_html(index=False, classes="scomp-table").replace('border="1"', 'border="0"')
                tab = tab.replace('class="dataframe scomp-table"', f'id="{table_id}" class="scomp-table"')
                tab = tab.replace('style="text-align: right;"', 'style="text-align: left;"')

            self.html_report += f"""
            <div id="table-wrapper">
                <div id="table-scroll">
                    {tab}
                </div>
            </div>
            <style>
                .scomp-table {{
                    width: 100%;
                    border-collapse: collapse;
                    font-size: 13px;
                    font-family: inherit;
                }}
                .scomp-table thead tr {{
                    background-color: {self.main_color} !important;
                    color: white !important;
                    text-align: left;
                }}
                .scomp-table th {{
                    color: white !important;
                    background-color: {self.main_color} !important;
                }}
                .scomp-table th, .scomp-table td {{
                    padding: 10px 12px;
                    border-bottom: 1px solid #e0e0e0;
                }}
                .scomp-table tbody tr:nth-child(even) {{
                    background-color: #f8f9fa;
                }}
                .scomp-table tbody tr:hover {{
                    background-color: {self.light_color}22;
                }}
            </style>"""
            logger.info("Added table to report!")
        else:
            logger.info("The DataFrame is to big!")


    # ═══════════════════════════════════════════════════════════════════
    # Advanced report components
    # ═══════════════════════════════════════════════════════════════════

    def add_kpi_cards(self, metrics: dict, cols: int = 3) -> None:
        """Add a row of KPI summary cards to the report.

        Displays key metrics as colored cards with optional trend indicators.

        :param metrics: dict mapping metric names to their config. Each value can be:
            - str/number: displayed as-is (no coloring)
            - dict with keys:
                - "value" (required): the displayed value
                - "trend" (optional): trend text (e.g. "+1.3%", "-0.5")
                - "status" (optional): "good" | "warning" | "critical" → card border color
                - "subtitle" (optional): small text below the value
        :param cols: number of columns in the grid (default 3)

        ## example
        report.add_kpi_cards({
            "Accuracy": {"value": "94.2%", "trend": "+1.3%", "status": "good"},
            "RMSE": {"value": "0.087", "status": "good"},
            "Latency": {"value": "230ms", "trend": "+15ms", "status": "warning"},
            "Drift Score": {"value": "0.31", "status": "critical"},
            "Samples": {"value": "12,450", "subtitle": "last 24h"},
        })
        """
        status_colors = {
            "good": "var(--accent3)",       # green
            "warning": "var(--accent4)",    # orange
            "critical": "var(--accent5)",   # red/pink
        }
        trend_colors = {
            "good": "#0f9d58",
            "warning": "#e8590c",
            "critical": "#d6336c",
        }

        html = f'<div style="display:grid;grid-template-columns:repeat({cols},1fr);gap:1rem;margin:1rem 0;">'

        for name, config in metrics.items():
            if not isinstance(config, dict):
                config = {"value": str(config)}

            value = config.get("value", "")
            trend = config.get("trend", "")
            status = config.get("status", "")
            subtitle = config.get("subtitle", "")

            border_color = status_colors.get(status, "var(--border)")
            trend_color = trend_colors.get(status, "var(--dim)")

            # Trend arrow
            trend_html = ""
            if trend:
                arrow = "↑" if trend.startswith("+") else "↓" if trend.startswith("-") else ""
                trend_html = (
                    f'<span style="font-size:.8rem;color:{trend_color};font-weight:600;'
                    f'margin-left:.5rem;">{arrow} {trend}</span>'
                )

            subtitle_html = ""
            if subtitle:
                subtitle_html = f'<div style="font-size:.7rem;color:var(--dim);margin-top:.15rem;">{subtitle}</div>'

            html += f"""<div style="background:var(--card);border:2px solid {border_color};
                border-radius:var(--radius);padding:1.2rem 1rem;text-align:center;">
                <div style="font-size:.75rem;color:var(--dim);text-transform:uppercase;
                    letter-spacing:.04em;font-weight:600;margin-bottom:.4rem;">{name}</div>
                <div style="font-size:1.6rem;font-weight:800;color:var(--text);">
                    {value}{trend_html}
                </div>
                {subtitle_html}
            </div>"""

        html += "</div>"
        self.html_report += html
        logger.info("Added KPI cards to report!")

    def add_plotly_grid(
        self,
        figures: list,
        cols: int = 2,
        titles: list[str] | None = None,
        height: int | None = None,
    ) -> None:
        """Add multiple Plotly figures arranged in a CSS grid.

        Each figure is rendered independently (not as Plotly subplots) and
        arranged in a responsive grid layout.

        :param figures: list of Plotly Figure objects
        :param cols: number of columns (default 2, responsive down to 1 on mobile)
        :param titles: optional list of titles for each figure (same length as figures)
        :param height: optional fixed height in px for each chart container

        ## example
        report.add_plotly_grid([fig1, fig2, fig3, fig4], cols=2, titles=["A", "B", "C", "D"])
        """
        import plotly.io as pio

        height_style = f"min-height:{height}px;" if height else "min-height:300px;"

        html = (
            f'<div style="display:grid;grid-template-columns:repeat({cols},1fr);'
            f'gap:1rem;margin:1rem 0;">'
        )

        for i, fig in enumerate(figures):
            title = titles[i] if titles and i < len(titles) else ""
            fig_html = pio.to_html(
                fig, include_plotlyjs=False, full_html=False,
                config={"responsive": True},
            )
            title_html = f'<h3 style="margin-bottom:.5rem;font-size:.9rem;">{title}</h3>' if title else ""
            html += (
                f'<div style="background:var(--card);border:1px solid var(--border);'
                f'border-radius:var(--radius);padding:1rem;{height_style}overflow:hidden;">'
                f'{title_html}{fig_html}</div>'
            )

        html += "</div>"

        # Add responsive media query for single column on small screens
        html += f"""<style>
            @media(max-width:768px){{
                div[style*="grid-template-columns:repeat({cols}"] {{
                    grid-template-columns:1fr !important;
                }}
            }}
        </style>"""

        self.html_report += html
        logger.info("Added plotly grid to report!")

    def add_tabs(self, tabs: dict, title: str = "") -> None:
        """Add tabbed content navigation to the report.

        Displays horizontal tabs that switch visible content on click.
        Content can be HTML strings or Plotly Figures.

        :param tabs: dict mapping tab labels to content (str, Plotly Figure, or pd.DataFrame)
        :param title: optional section title above the tabs

        ## example
        report.add_tabs({
            "Overview": "<p>Summary text here</p>",
            "Chart": plotly_figure,
            "Data": dataframe,
        }, title="Results")
        """
        import plotly.io as pio

        uid = uuid.uuid4().hex[:8]

        # Build tab buttons and content panels
        buttons_html = f'<div id="tabs_{uid}" style="display:flex;gap:0;border-bottom:2px solid var(--border);margin:1rem 0 0;">'
        panels_html = ""

        for i, (label, content) in enumerate(tabs.items()):
            active = i == 0
            safe_label = label.replace(" ", "_").replace("-", "_")
            panel_id = f"tabpanel_{uid}_{safe_label}"

            # Tab button
            active_style = (
                f"border-bottom:3px solid {self.main_color};color:{self.main_color};font-weight:700;"
                if active else "border-bottom:3px solid transparent;color:var(--dim);"
            )
            buttons_html += (
                f'<button onclick="switchTab_{uid}(\'{safe_label}\')" '
                f'id="tabbtn_{uid}_{safe_label}" '
                f'style="padding:.6rem 1.2rem;background:none;border:none;cursor:pointer;'
                f'font-size:.85rem;transition:all .2s;{active_style}">{label}</button>'
            )

            # Panel content
            display = "block" if active else "none"

            # Convert content to HTML (check Plotly Figure first — it also has to_html)
            if hasattr(content, "data") and hasattr(content, "layout"):  # Plotly Figure
                content_html = pio.to_html(
                    content, include_plotlyjs=False, full_html=False,
                    config={"responsive": True},
                )
            elif hasattr(content, "to_html") and hasattr(content, "columns"):  # DataFrame
                content_html = content.to_html(index=False, classes="scomp-table")
            else:
                content_html = str(content)

            panels_html += (
                f'<div id="{panel_id}" style="display:{display};padding:1rem 0;">'
                f'{content_html}</div>'
            )

        buttons_html += "</div>"

        # JS for tab switching
        script = f"""<script>
function switchTab_{uid}(tab) {{
    document.querySelectorAll('[id^="tabpanel_{uid}_"]').forEach(function(p) {{ p.style.display = 'none'; }});
    document.querySelectorAll('[id^="tabbtn_{uid}_"]').forEach(function(b) {{
        b.style.borderBottom = '3px solid transparent'; b.style.color = 'var(--dim)'; b.style.fontWeight = '400';
    }});
    var panel = document.getElementById('tabpanel_{uid}_' + tab);
    var btn = document.getElementById('tabbtn_{uid}_' + tab);
    if (panel) {{ panel.style.display = 'block'; }}
    if (btn) {{ btn.style.borderBottom = '3px solid {self.main_color}'; btn.style.color = '{self.main_color}'; btn.style.fontWeight = '700'; }}
    // Resize Plotly charts in newly visible panel
    if (panel && window.Plotly) {{
        setTimeout(function() {{
            panel.querySelectorAll('.js-plotly-plot').forEach(function(p) {{ Plotly.Plots.resize(p); }});
        }}, 100);
    }}
}}
</script>"""

        title_html = f"<h2>{title}</h2>" if title else ""
        self.html_report += title_html + buttons_html + panels_html + script
        logger.info("Added tabs to report!")

    def add_comparison_table(
        self,
        df,
        baseline_col: str,
        compare_cols: list[str],
        metric_col: str | None = None,
        higher_is_better: dict[str, bool] | None = None,
    ) -> None:
        """Add a comparison table with delta indicators between columns.

        Shows a baseline column and comparison columns with color-coded deltas (Δ).
        Useful for model comparison or A/B test results.

        :param df: pandas or polars DataFrame with metrics as rows
        :param baseline_col: column name to use as the reference
        :param compare_cols: list of column names to compare against baseline
        :param metric_col: column containing metric names (used as row labels). If None, uses the index.
        :param higher_is_better: dict mapping metric names to bool. If not provided,
            assumes higher is better for all. Used to determine arrow/color direction.

        ## example
        df = pd.DataFrame({
            "metric": ["accuracy", "rmse", "latency_ms"],
            "model_v1": [0.92, 0.12, 45],
            "model_v2": [0.95, 0.09, 52],
            "model_v3": [0.94, 0.10, 38],
        })
        report.add_comparison_table(df, baseline_col="model_v1",
            compare_cols=["model_v2", "model_v3"], metric_col="metric",
            higher_is_better={"accuracy": True, "rmse": False, "latency_ms": False})
        """
        import pandas as pd

        if hasattr(df, "to_pandas"):
            df = df.to_pandas()

        higher_is_better = higher_is_better or {}

        # Determine metric labels
        if metric_col and metric_col in df.columns:
            metrics = df[metric_col].tolist()
            value_df = df.drop(columns=[metric_col])
        else:
            metrics = [str(i) for i in df.index]
            value_df = df

        # Build table
        html = '<div style="overflow-x:auto;margin:1rem 0;">'
        html += '<table style="width:100%;border-collapse:collapse;font-size:.85rem;font-family:inherit;">'

        # Header
        html += '<thead><tr style="border-bottom:2px solid var(--border);">'
        html += '<th style="padding:.6rem;text-align:left;color:var(--dim);font-size:.75rem;text-transform:uppercase;">Metric</th>'
        html += f'<th style="padding:.6rem;text-align:center;color:var(--dim);font-size:.75rem;text-transform:uppercase;">{baseline_col}<br><span style="font-size:.65rem;font-weight:400;">(baseline)</span></th>'
        for col in compare_cols:
            html += f'<th style="padding:.6rem;text-align:center;color:var(--dim);font-size:.75rem;text-transform:uppercase;">{col}<br><span style="font-size:.65rem;font-weight:400;">vs baseline</span></th>'
        html += '</tr></thead><tbody>'

        # Rows
        for i, metric_name in enumerate(metrics):
            html += '<tr style="border-bottom:1px solid var(--border);">'
            html += f'<td style="padding:.5rem;font-weight:600;">{metric_name}</td>'

            baseline_val = value_df[baseline_col].iloc[i]
            html += f'<td style="padding:.5rem;text-align:center;font-family:monospace;">{self._format_num(baseline_val)}</td>'

            hib = higher_is_better.get(metric_name, True)

            for col in compare_cols:
                comp_val = value_df[col].iloc[i]
                delta = comp_val - baseline_val

                if pd.isna(delta) or pd.isna(comp_val):
                    html += '<td style="padding:.5rem;text-align:center;color:var(--dim);">—</td>'
                    continue

                # Determine if delta is "good" or "bad"
                is_improvement = (delta > 0 and hib) or (delta < 0 and not hib)
                is_worse = (delta < 0 and hib) or (delta > 0 and not hib)

                if is_improvement:
                    color = "#0f9d58"
                    arrow = "↑" if delta > 0 else "↓"
                elif is_worse:
                    color = "#d6336c"
                    arrow = "↓" if delta < 0 else "↑"
                else:
                    color = "var(--dim)"
                    arrow = "="

                delta_str = f"+{delta:.4g}" if delta > 0 else f"{delta:.4g}"
                html += (
                    f'<td style="padding:.5rem;text-align:center;">'
                    f'<span style="font-family:monospace;">{self._format_num(comp_val)}</span> '
                    f'<span style="font-size:.75rem;color:{color};font-weight:600;">'
                    f'{arrow} {delta_str}</span></td>'
                )

            html += '</tr>'

        html += '</tbody></table></div>'
        self.html_report += html
        logger.info("Added comparison table to report!")

    def add_summary_stats(self, df, title: str = "Data Summary") -> None:
        """Add an auto-generated data profiling summary to the report.

        Displays a compact table with column statistics: type, non-null count,
        missing %, unique values, and distribution indicators.

        :param df: pandas or polars DataFrame to profile
        :param title: section title (default "Data Summary")

        ## example
        report.add_summary_stats(df, title="Training Data Overview")
        """
        import pandas as pd

        if hasattr(df, "to_pandas"):
            df = df.to_pandas()

        n_rows = len(df)
        html = f'<h2>{title}</h2>'
        html += f'<p style="color:var(--dim);font-size:.85rem;margin-bottom:.75rem;">{n_rows:,} rows × {len(df.columns)} columns</p>'
        html += '<div style="overflow-x:auto;">'
        html += '<table style="width:100%;border-collapse:collapse;font-size:.8rem;font-family:inherit;">'

        # Header
        html += '<thead><tr style="background:var(--card);border-bottom:2px solid var(--border);">'
        for col_header in ["Column", "Type", "Non-Null", "Missing %", "Unique", "Sample Values"]:
            html += f'<th style="padding:.5rem .6rem;text-align:left;color:var(--dim);font-size:.7rem;text-transform:uppercase;letter-spacing:.03em;">{col_header}</th>'
        html += '</tr></thead><tbody>'

        for col in df.columns:
            series = df[col]
            dtype = str(series.dtype)
            non_null = int(series.notna().sum())
            missing_pct = (1 - non_null / n_rows) * 100 if n_rows > 0 else 0
            n_unique = int(series.nunique())

            # Sample values (first 3 unique non-null values)
            unique_vals = series.dropna().unique()[:3]
            sample_str = ", ".join(str(v)[:20] for v in unique_vals)
            if len(sample_str) > 50:
                sample_str = sample_str[:50] + "…"

            # Missing color
            if missing_pct == 0:
                miss_color = "var(--accent3)"
            elif missing_pct < 5:
                miss_color = "var(--text)"
            elif missing_pct < 20:
                miss_color = "var(--accent4)"
            else:
                miss_color = "var(--accent5)"

            # Type badge color
            if "int" in dtype or "float" in dtype:
                type_color = "var(--accent)"
            elif "object" in dtype or "str" in dtype:
                type_color = "var(--accent2)"
            elif "datetime" in dtype or "date" in dtype:
                type_color = "var(--accent3)"
            elif "bool" in dtype:
                type_color = "var(--accent4)"
            else:
                type_color = "var(--dim)"

            html += '<tr style="border-bottom:1px solid var(--border);">'
            html += f'<td style="padding:.4rem .6rem;font-weight:600;">{col}</td>'
            html += f'<td style="padding:.4rem .6rem;"><code style="color:{type_color};font-size:.75rem;">{dtype}</code></td>'
            html += f'<td style="padding:.4rem .6rem;font-family:monospace;">{non_null:,}</td>'
            html += f'<td style="padding:.4rem .6rem;color:{miss_color};font-weight:600;">{missing_pct:.1f}%</td>'
            html += f'<td style="padding:.4rem .6rem;font-family:monospace;">{n_unique:,}</td>'
            html += f'<td style="padding:.4rem .6rem;color:var(--dim);font-size:.75rem;">{sample_str}</td>'
            html += '</tr>'

        html += '</tbody></table></div>'
        self.html_report += html
        logger.info("Added summary stats to report!")

    def add_dark_mode_toggle(self) -> None:
        """Add a dark/light mode toggle button to the report.

        Inserts a floating toggle button in the top-right corner that switches
        between light mode (default) and dark mode by swapping CSS custom properties.

        ## example
        report.add_dark_mode_toggle()
        """
        uid = uuid.uuid4().hex[:8]
        html = f"""
        <button id="darkToggle_{uid}" onclick="toggleDarkMode_{uid}()"
            style="position:fixed;top:1rem;right:1rem;z-index:9999;
            background:var(--card);border:1px solid var(--border);border-radius:50%;
            width:40px;height:40px;cursor:pointer;font-size:1.2rem;
            display:flex;align-items:center;justify-content:center;
            box-shadow:0 2px 8px rgba(0,0,0,.1);transition:all .3s;">🌙</button>
        <script>
        (function() {{
            var isDark = false;
            var root = document.documentElement;
            var btn = document.getElementById('darkToggle_{uid}');
            var lightVars = {{
                '--bg': '#ffffff', '--card': '#f8fafc', '--text': '#1e293b',
                '--dim': '#64748b', '--border': '#e2e8f0',
            }};
            var darkVars = {{
                '--bg': '#0f172a', '--card': '#1e293b', '--text': '#e2e8f0',
                '--dim': '#94a3b8', '--border': '#334155',
            }};
            window.toggleDarkMode_{uid} = function() {{
                isDark = !isDark;
                var vars = isDark ? darkVars : lightVars;
                Object.keys(vars).forEach(function(k) {{ root.style.setProperty(k, vars[k]); }});
                btn.textContent = isDark ? '☀️' : '🌙';
                // Update body and report backgrounds
                document.body.style.background = vars['--bg'];
                var report = document.querySelector('.report');
                if (report) report.style.background = vars['--bg'];
                // Update tables
                document.querySelectorAll('.scomp-table thead tr, .scomp-table th').forEach(function(el) {{
                    if (isDark) {{ el.style.backgroundColor = '#334155'; }}
                    else {{ el.style.backgroundColor = ''; }}
                }});
            }};
        }})();
        </script>"""
        self.html_report += html
        logger.info("Added dark mode toggle to report!")

    @staticmethod
    def _format_num(val) -> str:
        """Format a numeric value for table display."""
        import pandas as pd
        if pd.isna(val):
            return "—"
        if isinstance(val, float):
            if abs(val) < 0.01 or abs(val) >= 10000:
                return f"{val:.4g}"
            return f"{val:.4f}".rstrip("0").rstrip(".")
        return str(val)

    def save_pdf(self, file_name="export.pdf"):
        """
        Saves the report as a PDF by rendering the HTML in a headless browser.
        This ensures all JavaScript (Plotly/Highcharts) is executed and visible.
        Automatically installs Chromium on first use if not already present.

        :param file_name: str - output PDF file path (default 'export.pdf')

        ## example
        report.save_pdf('my_report.pdf')
        """
        import subprocess
        import tempfile

        from playwright.sync_api import sync_playwright

        # Auto-install Chromium if not present
        try:
            with sync_playwright() as p:
                p.chromium.executable_path
        except Exception:
            logger.info("Chromium not found. Installing automatically...")
            subprocess.run(["playwright", "install", "chromium"], check=True)

        fd, temp_html_path = tempfile.mkstemp(suffix=".html")
        os.close(fd)
        self.save_html(temp_html_path)

        logger.info("Starting PDF generation... Loading graphs.")

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                file_url = f"file://{os.path.abspath(temp_html_path)}"
                page.goto(file_url, wait_until="networkidle")
                page.wait_for_timeout(2000)
                # Prepare page for PDF: fix layout, avoid page breaks
                page.evaluate("""() => {
                    // Page break avoidance
                    document.querySelectorAll('.plotly-graph-div, img, #table-wrapper, .print-grid-item').forEach(el => {
                        el.style.pageBreakInside = 'avoid';
                        el.style.breakInside = 'avoid';
                    });
                    document.querySelectorAll('h2').forEach(el => {
                        el.style.pageBreakAfter = 'avoid';
                        el.style.breakAfter = 'avoid';
                    });
                    
                    // Hide UI elements (combo boxes, labels, buttons, select2 widgets)
                    document.querySelectorAll('select, input[type="submit"], input[type="button"], label, br, .form-control, .select2-container').forEach(el => {
                        el.style.display = 'none';
                        el.style.visibility = 'hidden';
                        el.style.height = '0';
                        el.style.overflow = 'hidden';
                    });
                    
                    // Open all collapsed sections
                    document.querySelectorAll('.content').forEach(el => {
                        el.style.display = 'block';
                    });
                    
                    // Grid: show all items, equal columns
                    document.querySelectorAll('.print-grid-container').forEach(container => {
                        container.style.display = 'grid';
                        container.style.gridTemplateColumns = '1fr 1fr';
                        container.style.gap = '20px';
                    });
                    document.querySelectorAll('.print-grid-item').forEach(el => {
                        el.style.display = 'block';
                    });
                    
                    // Constrain all Plotly graphs to container
                    document.querySelectorAll('.plotly-graph-div').forEach(el => {
                        el.style.maxWidth = '100%';
                        el.style.overflow = 'hidden';
                    });
                    
                    // Remove report padding to use full page width
                    var report = document.querySelector('.report');
                    if (report) {
                        report.style.paddingLeft = '5%';
                        report.style.paddingRight = '5%';
                    }
                }""")

                # Resize Plotly graphs to fit their containers
                page.wait_for_timeout(500)
                page.evaluate("""() => {
                    if (window.Plotly) {
                        // Get the report content width
                        var report = document.querySelector('.report');
                        var reportWidth = report ? report.clientWidth : 800;
                        
                        document.querySelectorAll('.js-plotly-plot').forEach(el => {
                            var container = el.closest('.plotly-graph-div');
                            var inGrid = el.closest('.print-grid-item');
                            if (container) {
                                var targetWidth = inGrid ? container.clientWidth : reportWidth;
                                Plotly.relayout(el, { width: targetWidth, autosize: true });
                            }
                        });
                    }
                }""")
                page.wait_for_timeout(1000)
                page.pdf(
                    path=file_name,
                    format="A4",
                    print_background=True,
                    scale=0.5,
                    margin={"top": "20px", "bottom": "20px", "left": "20px", "right": "20px"},
                )
                browser.close()
                logger.info(f"PDF successfully saved as {file_name}!")
        except Exception as e:
            logger.info(f"An error occurred while generating the PDF: {e}")
        finally:
            if os.path.exists(temp_html_path):
                os.remove(temp_html_path)

    def save_html(self, file_name="export.html"):
        js = """
                        <script>
                        $(".js-example-tags").select2({
                          tags: true
                        });
                        </script>"""
        html_txt = f"""<!DOCTYPE html>
                        <html lang="{self.lan}">
                            <head>
                            {self.html_meta_info}
                            {self.html_title}
                            {self.html_layout}
                            </head>
                        <body>    
                            {self.header}
                            <div class="report" style="background-color:WHITE">
                                {self.html_report}
                            </div>
                            {self.footer}
                            
                            {js}
                        </body>
                        </html>
                    """
        with open(file_name, "w", encoding="utf-8") as f:
            f.write(html_txt)
        logger.info("Saved!")
