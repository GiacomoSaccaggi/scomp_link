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
                            // Select rows from table_id
                            var rows = document.querySelectorAll('table#' + table_id + ' tr');
                            // Construct csv
                            var csv = [];
                            for (var i = 0; i < rows.length; i++) {
                                var row = [], cols = rows[i].querySelectorAll('td, th');
                                for (var j = 0; j < cols.length; j++) {
                                    // Clean innertext to remove multiple spaces and jumpline (break csv)
                                    var data = cols[j].innerText.replace(/(\\r\\n|\\n|\\r)/gm, '').replace(/(\\s\\s)/gm, ' ')
                                    // Escape double-quote with double-double-quote (see https://stackoverflow.com/questions/17808511/properly-escape-a-double-quote-in-csv)
                                    data = data.replace(/"/g, '""');
                                    // Push escaped string
                                    row.push('"' + data + '"');
                                }
                                csv.push(row.join(separator));
                            }
                            var csv_string = csv.join('\\n');
                            // Download it
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
                        var coll = document.getElementsByClassName("collapsiblemygs");
                        var i;

                        for (i = 0; i < coll.length; i++) {
                          coll[i].addEventListener("click", function() {
                            this.classList.toggle("active");
                            var content = this.nextElementSibling;
                            if (content.style.display === "block") {
                              content.style.display = "none";
                            } else {
                              content.style.display = "block";
                            }
                          });
                        }
                        document.addEventListener("DOMContentLoaded", function() {
                            // Function to resize svg-containers, SVG images and rect elements
                            function resizeElements() {
                                // Seleziona tutti gli elementi con la classe 'svg-container'
                                var svgContainers = document.querySelectorAll('.svg-container');
                        
                                // Itera su ogni svg-container e imposta la larghezza al 100%
                                svgContainers.forEach(function(container) {
                                    container.style.width = '100%';
                                });
                                
                                 var svgContainers0 = document.querySelectorAll('.user-select-none');
                        
                                // Itera su ogni svg-container e imposta la larghezza al 100%
                                svgContainers0.forEach(function(container) {
                                    container.style.width = '100%';
                                });
                                
                                
                                 var cartesianlayer = document.querySelectorAll('.cartesianlayer');
                        
                                // Itera su ogni svg-container e imposta la larghezza al 100%
                                cartesianlayer.forEach(function(container) {
                                    container.style.width = '100%';
                                });
                                
                                
                        
                                 var svgContainers1 = document.querySelectorAll('.js-plotly-plot');
                        
                                // Itera su ogni svg-container e imposta la larghezza al 100%
                                svgContainers1.forEach(function(container) {
                                    container.style.width = '100%';
                                });
                                
                                
                                 var svgContainers2 = document.querySelectorAll('.main-svg');
                        
                                // Itera su ogni svg-container e imposta la larghezza al 100%
                                svgContainers2.forEach(function(container) {
                                    container.style.width = '100%';
                                });
                        
                                 var svgContainers3 = document.querySelectorAll('.plot-container');
                        
                                // Itera su ogni svg-container e imposta la larghezza al 100%
                                svgContainers3.forEach(function(container) {
                                    container.style.width = '100%';
                                });
                        
                                // Seleziona tutti gli elementi con il tag '.highcharts-container'
                                var items = document.querySelectorAll('.highcharts-container');
                        
                                // Itera su ogni rect e imposta la larghezza al 100%
                                items.forEach(function(item) {
                                    item.style.width = '100%';
                                });
                                
                                
                                // Seleziona tutti gli elementi con il tag 'select'
                                var selects = document.querySelectorAll('select');
                        
                                // Itera su ogni rect e imposta la larghezza al 100%
                                selects.forEach(function(select) {
                                    select.style.width = '200px';
                                });
                                
                                
                                // Seleziona tutti gli elementi con il tag 'content'
                                var contents = document.querySelectorAll('.content');
                        
                                contents.forEach(function(content) {
                                    content.style.display = 'none';
                                });
                                
                                 
                            }
                        
                            // Call the function when the document is fully loaded
                            resizeElements();
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
            html,body{font-family:{font_family}, sans-serif;font-size:15px;line-height:1.5;text-align: justify;text-justify: inter-word;color:#4D577D;}html{overflow-x:hidden}
            h1{font-size:32px}h2{font-size:25px;color:#32BBB9;}h3{font-size:15px}h4{font-size:12px}h5{font-size:10px}h6{font-size:9px}
            h1,h2,h3,h4,h5,h6{font-weight:400;margin:10px 0}.w3-wide{letter-spacing:4px}
            footer{text-align: center;padding-left: 15%;padding-right: 15%;}
            header{
              position: relative;
              background-position: center;
              background-repeat: no-repeat;
              background-size: cover;
              background-image: url({url_background_header});
              height: 50%;
              width:100%;
              color: white;
              text-align: left;
              padding-top: 5%;
              padding-left: 15%;
            }
            .user-select-none .svg-container {
              height: 100%;
              width:100%;
            }
            .report {
                padding-right: 20%;
                padding-bottom: 5%;
                padding-left: 20%;
            }
            input[type=button], input[type=submit], input[type=reset] {
              padding: 16px 32px;
              text-decoration: none;
              margin: 4px 2px;
              cursor: pointer;
              display: inline-block;
              border-radius: 4px;
              background-color: white;
              color:  {dark_color};
              border: 2px solid  {main_color};
              text-align: center;
              transition: all 0.5s;
              cursor: pointer;
              margin: 5px;
            }
           :root {
              --radius: 2px;
              --baseFg: dimgray;
              --baseBg: white;
              --accentFg: #5528D2;
              --accentBg:  {main_color};
            }
            input[type=button]:hover, input[type=submit]:hover, input[type=reset]:hover {
                  background-color:  {main_color};
                  color: white;
                }

            .plotly-graph-div {
                        width: 100%;
            }    
            .highcharts-label-icon {
                opacity: 0.5;
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
            #table-wrapper {
              position:relative;
            }
            #table-scroll {
              height:500px;
              overflow:auto;
              margin-top:20px;
            }
            #table-wrapper table {
              width:100%;
            
            }
            #table-wrapper table * {
              color:black;
            }
            #table-wrapper table thead th .text {
              position:absolute;
              top:-20px;
              z-index:2;
              height:20px;
              width:35%;
              border:1px solid red;
            }
            .column_result_save {
              float: left;
              width: 50%;
              padding: 10px;
            }

            select {
              font: 400 12px/1.3 sans-serif;
              -webkit-appearance: none;
              appearance: none;
              color: var(--baseFg);
              border: 1px solid var(--baseFg);
              line-height: 1;
              outline: 0;
              padding: 0.65em 2.5em 0.55em 0.75em;
              border-radius: var(--radius);
              background-color: var(--baseBg);
              background-image: linear-gradient(var(--baseFg), var(--baseFg)),
                linear-gradient(-135deg, transparent 50%, var(--accentBg) 50%),
                linear-gradient(-225deg, transparent 50%, var(--accentBg) 50%),
                linear-gradient(var(--accentBg) 42%, var(--accentFg) 42%);
              background-repeat: no-repeat, no-repeat, no-repeat, no-repeat;
              background-size: 1px 100%, 20px 22px, 20px 22px, 20px 100%;
              background-position: right 20px center, right bottom, right bottom, right bottom;   
            }

            select:hover {
              background-image: linear-gradient(var(--accentFg), var(--accentFg)),
                linear-gradient(-135deg, transparent 50%, var(--accentFg) 50%),
                linear-gradient(-225deg, transparent 50%, var(--accentFg) 50%),
                linear-gradient(var(--accentFg) 42%, var(--accentBg) 42%);
            }

            select:active {
              background-image: linear-gradient(var(--accentFg), var(--accentFg)),
                linear-gradient(-135deg, transparent 50%, var(--accentFg) 50%),
                linear-gradient(-225deg, transparent 50%, var(--accentFg) 50%),
                linear-gradient(var(--accentFg) 42%, var(--accentBg) 42%);
              color: var(--accentBg);
              border-color: var(--accentFg);
              background-color: var(--accentFg);
            }
            .collapsiblemygs {
                  background-color: #777;
                  color: white;
                  cursor: pointer;
                  padding: 18px;
                  width: 100%;
                  border: none;
                  text-align: left;
                  outline: none;
                  font-size: 15px;
                }

                .active, .collapsiblemygs:hover {
                  background-color: #555;
                }

                .content {
                  padding: 0 18px;
                  display: block;
                  overflow: hidden;
                  background-color: #f1f1f1;
                }
            /*rules*/
            @media only screen and (max-width: 1024px) {
            .report{padding-right: 5%; padding-left: 5%;}
            }
            /* ----------- iPhone 6+, 7+ and 8+ ----------- */
            /* Portrait */
            @media only screen and (max-width: 1024px) {
            h2{font-size:18px}
            h3{font-size:10px}
            }
            @media only screen and (max-width: 835px) {
            h2{font-size:18px}
            h3{font-size:8px}
            }
            @media only screen and (max-width: 700px) {
            h2{font-size:15px}
            h3{font-size:6px}
            }
            @media only screen and (max-width: 525px) {
            h2{font-size:13px}
            h3{font-size:5px}
            }
            @media only screen and (max-width: 370px) {
            h2{font-size:10px}
            h3{font-size:3px}
            }
            /* --- PDF PRINT STYLES --- */
            @media print {
                .collapsiblemygs {
                    page-break-before: always;
                    background-color: transparent !important;
                    color: black !important;
                    font-size: 24px !important;
                    font-weight: bold;
                    border-bottom: 2px solid var(--accentBg);
                    padding: 0 0 10px 0 !important;
                    margin-top: 30px !important;
                }
                .content {
                    display: block !important;
                    background-color: white !important;
                    padding: 0 !important;
                }
                select, input[type="submit"], input[type="button"], label {
                    display: none !important;
                }
                .print-grid-container {
                    display: flex !important;
                    flex-wrap: wrap !important;
                    justify-content: space-between !important;
                }
                .print-grid-item {
                    display: block !important;
                    width: 48% !important;
                    page-break-inside: avoid;
                }
                body {
                    -webkit-print-color-adjust: exact;
                    print-color-adjust: exact;
                }
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
            html,body{font-family:{font_family}, sans-serif;font-size:15px;line-height:1.5;text-align: justify;text-justify: inter-word;color:#4D577D;}html{overflow-x:hidden}
            h1{font-size:32px}h2{font-size:25px;color:#32BBB9;}h3{font-size:15px}h4{font-size:12px}h5{font-size:10px}h6{font-size:9px}
            h1,h2,h3,h4,h5,h6{font-weight:400;margin:10px 0}.w3-wide{letter-spacing:4px}
            footer{text-align: center;padding-left: 15%;padding-right: 15%;}
            header{
              position: relative;
              background-position: center;
              background-repeat: no-repeat;
              background-size: cover;
              background-image: url({url_background_header});
              height: 50%;
              width:100%;
              color: white;
              text-align: left;
              padding-top: 5%;
              padding-left: 15%;
            }
            .user-select-none .svg-container {
              height: 100%;
              width:100%;
            }
            .report {
                padding-right: 20%;
                padding-bottom: 5%;
                padding-left: 20%;
            }
            input[type=button], input[type=submit], input[type=reset] {
              padding: 16px 32px;
              text-decoration: none;
              margin: 4px 2px;
              cursor: pointer;
              display: inline-block;
              border-radius: 4px;
              background-color: white;
              color:  {dark_color};
              border: 2px solid  {main_color};
              text-align: center;
              transition: all 0.5s;
              cursor: pointer;
              margin: 5px;
            }
           :root {
              --radius: 2px;
              --baseFg: dimgray;
              --baseBg: white;
              --accentFg: #5528D2;
              --accentBg:  {main_color};
            }
            input[type=button]:hover, input[type=submit]:hover, input[type=reset]:hover {
                  background-color:  {main_color};
                  color: white;
                }

            .plotly-graph-div {
                        width: 100%;
            }
             #container {
                height: 600px;
            }
            .highcharts-label-icon {
                opacity: 0.5;
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
            #table-wrapper {
              position:relative;
            }
            #table-scroll {
              height:500px;
              overflow:auto;
              margin-top:20px;
            }
            #table-wrapper table {
              width:100%;
            
            }
            #table-wrapper table * {
              color:black;
            }
            #table-wrapper table thead th .text {
              position:absolute;
              top:-20px;
              z-index:2;
              height:20px;
              width:35%;
              border:1px solid red;
            }
            .column_result_save {
              float: left;
              width: 50%;
              padding: 10px;
            }
            select {
              font: 400 12px/1.3 sans-serif;
              -webkit-appearance: none;
              appearance: none;
              color: var(--baseFg);
              border: 1px solid var(--baseFg);
              line-height: 1;
              outline: 0;
              padding: 0.65em 2.5em 0.55em 0.75em;
              border-radius: var(--radius);
              background-color: var(--baseBg);
              background-image: linear-gradient(var(--baseFg), var(--baseFg)),
                linear-gradient(-135deg, transparent 50%, var(--accentBg) 50%),
                linear-gradient(-225deg, transparent 50%, var(--accentBg) 50%),
                linear-gradient(var(--accentBg) 42%, var(--accentFg) 42%);
              background-repeat: no-repeat, no-repeat, no-repeat, no-repeat;
              background-size: 1px 100%, 20px 22px, 20px 22px, 20px 100%;
              background-position: right 20px center, right bottom, right bottom, right bottom;   
            }

            select:hover {
              background-image: linear-gradient(var(--accentFg), var(--accentFg)),
                linear-gradient(-135deg, transparent 50%, var(--accentFg) 50%),
                linear-gradient(-225deg, transparent 50%, var(--accentFg) 50%),
                linear-gradient(var(--accentFg) 42%, var(--accentBg) 42%);
            }

            select:active {
              background-image: linear-gradient(var(--accentFg), var(--accentFg)),
                linear-gradient(-135deg, transparent 50%, var(--accentFg) 50%),
                linear-gradient(-225deg, transparent 50%, var(--accentFg) 50%),
                linear-gradient(var(--accentFg) 42%, var(--accentBg) 42%);
              color: var(--accentBg);
              border-color: var(--accentFg);
              background-color: var(--accentFg);
            }
            .collapsiblemygs {
                  background-color: #777;
                  color: white;
                  cursor: pointer;
                  padding: 18px;
                  width: 100%;
                  border: none;
                  text-align: left;
                  outline: none;
                  font-size: 15px;
                }

                .active, .collapsiblemygs:hover {
                  background-color: #555;
                }

                .content {
                  padding: 0 18px;
                  display: block;
                  overflow: hidden;
                  background-color: #f1f1f1;
                }
            /*rules*/
            @media only screen and (max-width: 1024px) {
            .report{padding-right: 5%; padding-left: 5%;}
            }
            /* ----------- iPhone 6+, 7+ and 8+ ----------- */
            /* Portrait */
            @media only screen and (max-width: 1024px) {
            h2{font-size:18px}
            h3{font-size:10px}
            }
            @media only screen and (max-width: 835px) {
            h2{font-size:18px}
            h3{font-size:8px}
            }
            @media only screen and (max-width: 700px) {
            h2{font-size:15px}
            h3{font-size:6px}
            }
            @media only screen and (max-width: 525px) {
            h2{font-size:13px}
            h3{font-size:5px}
            }
            @media only screen and (max-width: 370px) {
            h2{font-size:10px}
            h3{font-size:3px}
            }
            /* --- PDF PRINT STYLES --- */
            @media print {
                .collapsiblemygs {
                    page-break-before: always;
                    background-color: transparent !important;
                    color: black !important;
                    font-size: 24px !important;
                    font-weight: bold;
                    border-bottom: 2px solid var(--accentBg);
                    padding: 0 0 10px 0 !important;
                    margin-top: 30px !important;
                }
                .content {
                    display: block !important;
                    background-color: white !important;
                    padding: 0 !important;
                }
                select, input[type="submit"], input[type="button"], label {
                    display: none !important;
                }
                .print-grid-container {
                    display: flex !important;
                    flex-wrap: wrap !important;
                    justify-content: space-between !important;
                }
                .print-grid-item {
                    display: block !important;
                    width: 48% !important;
                    page-break-inside: avoid;
                }
                body {
                    -webkit-print-color-adjust: exact;
                    print-color-adjust: exact;
                }
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

    def single_plotly(
        self, fig: "plotly.graph_objs._figure.Figure", title: str, plotdivid: str = None
    ) -> "html plotly code":
        """
        This function it is usefull to create a single plot in html
        :type plotdivid: object
        """
        fig_dict = json.loads(to_json_plotly(fig))
        jdata = to_json_plotly(fig_dict.get("data", []))
        jlayout = to_json_plotly(fig_dict.get("layout", {}))
        jconfig = to_json_plotly({"responsive": True})
        if plotdivid is None:
            plotdivid = title.replace(" ", "_")
            for p in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~":
                plotdivid = plotdivid.replace(p, "_")

        options = (
            """ 
                // Imposta la larghezza al 100%
                Plotly.update('"""
            + plotdivid
            + """', { 'layout.width': window.innerWidth * 0.9 }); // Puoi anche usare '100%' al posto di 'window.innerWidth * 0.9' se preferisci una larghezza fissa
            
                // Aggiungi un listener per aggiornare la larghezza quando la finestra viene ridimensionata
                window.addEventListener('resize', function() {
                    Plotly.update('"""
            + plotdivid
            + """', { 'layout.width': window.innerWidth * 0.9 });
                });
                """
        )
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

    def select_plotly(self, figures_dict: dict, title: str, labels="Choose a label") -> "html plotly code":
        """

        :type figures_dict: object
        """
        import random

        option_val = []
        hide_element = ""
        script = ""
        multiple = type(list(figures_dict.keys())[0]) == tuple
        n_filters = len(list(figures_dict.keys())[0]) if multiple else 1
        if labels != "Choose a label" and len(labels) == n_filters:
            labels_ord = list(labels)
        else:
            labels_ord = ["Choose a label"] * n_filters
        title_ = title.replace(" ", "_")
        for p in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~":
            title_ = title_.replace(p, "_")
        if multiple:
            idhide_ = [f"{i}_{title_}" for i in range(n_filters)]
        else:
            idhide_ = [f"1_{title_}"]
        i = 0
        for single_title, fig in figures_dict.items():
            fig_dict = json.loads(to_json_plotly(fig))
            jdata = to_json_plotly(fig_dict.get("data", []))
            jlayout = to_json_plotly(fig_dict.get("layout", {}))
            jconfig = to_json_plotly({"responsive": True})
            plotdivid = "".join(single_title).replace(" ", "_")
            for p in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~":
                plotdivid = plotdivid.replace(p, "_")
            idhide = f"{title_}_{plotdivid}"
            hide_element += f'document.getElementById("{idhide}").style.display = "none"; '
            display = "block" if i == 0 else "none"
            i += 1
            # option_val.append([f'<option value="{idhide}">{single_title}</option>'])
            if multiple:
                option_val_tmp = []
                for k_part in single_title:
                    k_part_tmp = k_part.replace(" ", "_")
                    for p in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~":
                        k_part_tmp = k_part_tmp.replace(p, "_")
                    option_val_tmp.append(f'<option value="{k_part_tmp}">{k_part}</option>')
                option_val.append(option_val_tmp)
            else:
                single_title_tmp = single_title.replace(" ", "_")
                for p in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~":
                    single_title_tmp = single_title_tmp.replace(p, "_")
                option_val.append([f'<option value="{single_title_tmp}">{single_title}</option>'])
            id_plotdiv = f"plotdivid_{random.randint(1000, 9999)}"
            options = (
                """ 
                           // Imposta la larghezza al 100%
                           Plotly.update('"""
                + id_plotdiv
                + """', { 'layout.width': window.innerWidth * 0.9 }); // Puoi anche usare '100%' al posto di 'window.innerWidth * 0.9' se preferisci una larghezza fissa

                           // Aggiungi un listener per aggiornare la larghezza quando la finestra viene ridimensionata
                           window.addEventListener('resize', function() {
                               Plotly.update('"""
                + id_plotdiv
                + """', { 'layout.width': window.innerWidth * 0.9 });
                           });
                           """
            )
            script += """\
                    <div id="{idhide}" class="print-grid-item" style="display:{display}">
                        <h3 class="print-grid-title">{grid_title}</h3>
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
                    </div>
                        """.format(
                idhide=idhide,
                display=display,
                grid_title=" - ".join(single_title) if isinstance(single_title, tuple) else single_title,
                id=id_plotdiv,
                data=jdata,
                layout=jlayout,
                config=jconfig,
                options=options,
            )

        documentget = f"'{title_}_'+" + "+".join(
            [f'document.getElementById("labels{idhidepart}").value' for idhidepart in idhide_]
        ).replace(" ", "_")
        script = (
            f"<h2>{title}</h2>\n"
            + "".join(
                [
                    f"""
                           <label for="labels{idhidepart}">{lab}:</label>
                             <select name="labels{idhidepart}" id="labels{idhidepart}"  class="form-control js-example-tags">
                               {''.join(list(set([option_tmp[n] for option_tmp in option_val])))}
                             </select>
                         <br><br>"""
                    for n, [idhidepart, lab] in enumerate(zip(idhide_, labels_ord))
                ]
            )
            + f"""
                         <input type="submit"  onclick="SelectFunction{title_}()">
                   """
            + '<div class="print-grid-container">'
            + script
            + "</div>"
            + """
                        <script>
                            function SelectFunction{idhide_}() {start_fun}
                                {hide_element}
                                var value = {documentget};
                                document.getElementById(value).style.display = "block";
                            {end_fun}
                        </script>
                   """.format(
                idhide_=title_, documentget=documentget, hide_element=hide_element, start_fun="{", end_fun="}"
            )
        )
        return script

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

    def add_matplotlib_graph_to_report(
        self, fig: "matplotlib.figure.Figure", title: str, dpi: int = 150, img_format: str = "png"
    ):
        """
        Add a matplotlib figure to the report as a base64-encoded image.

        :param fig: matplotlib.figure.Figure
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

    def add_many_plots_with_selection_box_to_report(self, figures_dict: dict, title: str, **kwargs):
        """
        Add many graphs to report
        :param figures_dict: plotly.graph_objs._figure.Figure
        :param title: str
        :return:

        ## example
        demo_report = NielsenHTMLreport('My fisrt REPORT') # if you don't have just created
        import plotly.express as px
        fig1 = px.scatter(x=range(10), y=range(10))
        fig2 = px.scatter(x=range(20), y=range(20))
        figures_dict = {
                        'This is the first 1':fig1,
                        'This is the second 2':fig2
                        }
        demo_report.add_many_plots_with_selection_box_to_report(figures_dict, 'My first Graph')
        """
        labels = kwargs.get("labels", "Choose a label")
        self.html_report += self.select_plotly(figures_dict, title, labels=labels)
        logger.info("Added graph to report!")

    def open_section(self, section_title: str, ingore_multi_section=False) -> "html plotly code":
        if not self.section_just_open or ingore_multi_section:
            self.html_report += f'<button class="collapsiblemygs">{section_title}</button> <div class="content">'
            self.section_just_open = True
            logger.info("Open section to report!")
        else:
            logger.info("Warning you already have an open section")

    def close_section(self, ingore_multi_section=False) -> "html plotly code":
        if self.section_just_open or ingore_multi_section:
            self.html_report += "</div>"
            self.section_just_open = False
            logger.info("Close section to report!")
        else:
            logger.info("Warning you did not open section yet")

    def add_title(self, title: str) -> "html plotly code":
        self.html_report += f"<h2>{title}</h2>"
        logger.info("Added title to report!")

    def add_text(self, text: str) -> "html plotly code":
        self.html_report += f"<p>{text}</p>"
        logger.info("Added text to report!")

    def add_dataframe(self, df: pd.DataFrame, title: str, limit_max=2000) -> "html plotly code":
        if len(df) < limit_max:
            self.html_report += (
                '<a href="#" onclick="download_table_as_csv('
                + f"'{title.replace(' ', '')}'"
                + ');">Download as CSV</a>'
            )
            tab = df.to_html(index=False, classes="scomp-table").replace('border="1"', 'border="0"')
            tab = tab.replace('class="dataframe scomp-table"', f'id="{title.replace(" ", "")}" class="scomp-table"')
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
