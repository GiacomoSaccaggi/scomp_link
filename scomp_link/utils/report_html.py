# -*- coding: utf-8 -*-
# encoding utf-8
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

import pandas as pd
import plotly
import json
from plotly.io.json import to_json_plotly
# Read constants from encrypted file
import os
import jwt


class ScompLinkHTMLReport:
    def __init_subclass__(cls, title,
                         font_family='Baloo 2',
                         url_img_logo='',
                         # optional logo URL (empty for neutral branding)
                         # optional secondary logo URL (empty for neutral branding)
                         url_background_header='https://giacomosaccaggi.github.io/articoli/sfondo.png',
                         description='Automatic Report',
                         author='scomp-link toolkit',
                         language='en',
                         main_color='#6E37FA',
                         light_color='#9682FF',
                         dark_color='#4614B4'):
        cls.dark_color = dark_color
        cls.light_color = light_color
        cls.main_color = main_color
        cls.html_title = f'<title>{title}</title>'
        cls.header = f'''<header>
                                    <h1>{title}</h1><br><br><br>
                                </header>'''
        cls.html_meta_info = f"""
                    <meta charset="utf-8">
                    <meta http-equiv="X-UA-Compatible" content="IE=edge">
                    <meta name="viewport" content="width=device-width, initial-scale=1">
                    <meta name="description" content="{description}">
                    <meta name="author" content="{author}">
                    """
        cls.html_layout = """
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
            </style>
            <script src="https://cdn.plot.ly/plotly-2.9.0.min.js"></script>
        """.replace('{font_family}', font_family) \
            .replace('{url_img_logo}', url_img_logo) \
            .replace('{url_background_header}', url_background_header) \
            .replace('{main_color}', cls.main_color) \
            .replace('{dark_color}', cls.dark_color) \
            .replace('{light_color}', cls.light_color)
        cls.footer = '''
        
        
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
                                    var data = cols[j].innerText.replace(/(\\r\\n|\\n|\\r)/gm, '').replace(/(\s\s)/gm, ' ')
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
                            // Funzione per eseguire il resize degli svg-container, delle immagini SVG e dei rect
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
                        
                            // Chiama la funzione quando il documento è completamente caricato
                            resizeElements();
                        });
                        
                        </script>
                        </div><hr><br><footer><strong>About scomp-link</strong><br>
                        scomp-link is a general-purpose data science toolkit for preprocessing, model selection, and validation. It is dataset- and domain-agnostic.
                        Learn more in the project README and documentation.<br>
                        <strong>May the code be with you.</strong><br><br>
                        Copyright &copy; 2026 scomp-link contributors. All rights reserved.<br><br></footer>'''
        cls.html_report = ''
        cls.section_just_open = False
        cls.lan = language

    def __init__(self, title,
                font_family='Baloo 2',
                url_img_logo='',
                # optional logo URL (empty for neutral branding)
                # optional secondary logo URL (empty for neutral branding)
                url_background_header='',
                description='Automatic Report',
                author='scomp-link toolkit',
                language='en',
                main_color='#6E37FA',
                light_color='#9682FF',
                dark_color='#4614B4'):
        self.dark_color = dark_color
        self.light_color = light_color
        self.main_color = main_color
        self.html_title = f'<title>{title}</title>'
        self.header = f'''<header>
                            <h1>{title}</h1><br><br><br>
                        </header>'''
        self.html_meta_info = f"""
            <meta charset="utf-8">
            <meta http-equiv="X-UA-Compatible" content="IE=edge">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <meta name="description" content="{description}">
            <meta name="author" content="{author}">
            """
        self.html_layout = """
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
            </style>
            <script src="https://cdn.plot.ly/plotly-2.9.0.min.js"></script>
        """.replace('{font_family}', font_family) \
            .replace('{url_img_logo}', url_img_logo) \
            .replace('{url_background_header}', url_background_header) \
            .replace('{main_color}', self.main_color) \
            .replace('{dark_color}', self.dark_color) \
            .replace('{light_color}', self.light_color)
        self.footer = '''
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
                                    var data = cols[j].innerText.replace(/(\\r\\n|\\n|\\r)/gm, '').replace(/(\s\s)/gm, ' ')
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
                            // Funzione per eseguire il resize degli svg-container, delle immagini SVG e dei rect
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
                        
                            // Chiama la funzione quando il documento è completamente caricato
                            resizeElements();
                        });
                        
                        </script>
                        </div><hr><br><footer><strong>About scomp-link</strong><br>
                        scomp-link is a general-purpose data science toolkit for preprocessing, model selection, and validation. It is dataset- and domain-agnostic.
                        Learn more in the project README and documentation.<br>
                        <strong>May the code be with you.</strong><br><br>
                        Copyright &copy; 2026 scomp-link contributors. All rights reserved.<br><br></footer>'''
        self.html_report = ''
        self.section_just_open = False
        self.lan = language

    def single_plotly(self,
                      fig: 'plotly.graph_objs._figure.Figure',
                      title: str,
                      plotdivid: str = None
                      ) -> 'html plotly code':
        """
        This function it is usefull to create a single plot in html
        :type plotdivid: object
        """
        fig_dict = json.loads(to_json_plotly(fig))
        jdata = to_json_plotly(fig_dict.get("data", []))
        jlayout = to_json_plotly(fig_dict.get("layout", {}))
        jconfig = to_json_plotly({'responsive': True})
        if plotdivid is None:
            plotdivid = title.replace(' ', '_')
            for p in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~':
                plotdivid = plotdivid.replace(p, '_')

        options = """ 
                // Imposta la larghezza al 100%
                Plotly.update('"""+plotdivid+"""', { 'layout.width': window.innerWidth * 0.9 }); // Puoi anche usare '100%' al posto di 'window.innerWidth * 0.9' se preferisci una larghezza fissa
            
                // Aggiungi un listener per aggiornare la larghezza quando la finestra viene ridimensionata
                window.addEventListener('resize', function() {
                    Plotly.update('"""+plotdivid+"""', { 'layout.width': window.innerWidth * 0.9 });
                });
                """
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
                    """.format(
            title=title,
            id=plotdivid,
            data=jdata,
            layout=jlayout,
            config=jconfig,
            options=options
        )

        return script

    def select_plotly(self,
                      figures_dict: dict,
                      title: str,
                      labels='Choose a label'
                      ) -> 'html plotly code':
        """

        :type figures_dict: object
        """
        import random
        option_val = []
        hide_element = ''
        script = ''
        multiple = type(list(figures_dict.keys())[0]) == tuple
        n_filters = len(list(figures_dict.keys())[0]) if multiple else 1
        if labels != 'Choose a label' and len(labels) == n_filters:
            labels_ord = list(labels)
        else:
            labels_ord = ['Choose a label'] * n_filters
        title_ = title.replace(' ', '_')
        for p in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~':
            title_ = title_.replace(p, '_')
        if multiple:
            idhide_ = [f'{i}_{title_}' for i in range(n_filters)]
        else:
            idhide_ = [f'1_{title_}']
        i = 0
        for single_title, fig in figures_dict.items():
            fig_dict = json.loads(to_json_plotly(fig))
            jdata = to_json_plotly(fig_dict.get("data", []))
            jlayout = to_json_plotly(fig_dict.get("layout", {}))
            jconfig = to_json_plotly({'responsive': True})
            plotdivid = ''.join(single_title).replace(' ', '_')
            for p in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~':
                plotdivid = plotdivid.replace(p, '_')
            idhide = f'{title_}_{plotdivid}'
            hide_element += f'document.getElementById("{idhide}").style.display = "none"; '
            display = 'block' if i == 0 else 'none'
            i += 1
            # option_val.append([f'<option value="{idhide}">{single_title}</option>'])
            if multiple:
                option_val_tmp = []
                for k_part in single_title:
                    k_part_tmp = k_part.replace(" ", "_")
                    for p in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~':
                        k_part_tmp = k_part_tmp.replace(p, '_')
                    option_val_tmp.append(f'<option value="{k_part_tmp}">{k_part}</option>')
                option_val.append(option_val_tmp)
            else:
                single_title_tmp = single_title.replace(" ", "_")
                for p in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~':
                    single_title_tmp = single_title_tmp.replace(p, '_')
                option_val.append([f'<option value="{single_title_tmp}">{single_title}</option>'])
            id_plotdiv = f'plotdivid_{random.randint(1000, 9999)}'
            options = """ 
                           // Imposta la larghezza al 100%
                           Plotly.update('""" + id_plotdiv + """', { 'layout.width': window.innerWidth * 0.9 }); // Puoi anche usare '100%' al posto di 'window.innerWidth * 0.9' se preferisci una larghezza fissa

                           // Aggiungi un listener per aggiornare la larghezza quando la finestra viene ridimensionata
                           window.addEventListener('resize', function() {
                               Plotly.update('""" + id_plotdiv + """', { 'layout.width': window.innerWidth * 0.9 });
                           });
                           """
            script += """\
                    <div id="{idhide}" style="display:{display}">
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
                id=id_plotdiv,
                data=jdata,
                layout=jlayout,
                config=jconfig,
                options=options
            )

        documentget = f"'{title_}_'+" + '+'.join(
            [f'document.getElementById("labels{idhidepart}").value' for idhidepart in idhide_]).replace(' ', '_')
        script = f'<h2>{title}</h2>\n' + ''.join([f"""
                           <label for="labels{idhidepart}">{lab}:</label>
                             <select name="labels{idhidepart}" id="labels{idhidepart}"  class="form-control js-example-tags">
                               {''.join(list(set([option_tmp[n] for option_tmp in option_val])))}
                             </select>
                         <br><br>""" for n, [idhidepart, lab] in enumerate(zip(idhide_, labels_ord))]) \
                 + \
                 f"""
                         <input type="submit"  onclick="SelectFunction{title_}()">
                   """ + script + """
                        <script>
                            function SelectFunction{idhide_}() {start_fun}
                                {hide_element}
                                var value = {documentget};
                                document.getElementById(value).style.display = "block";
                            {end_fun}
                        </script>
                   """.format(
            idhide_=title_,
            documentget=documentget,
            hide_element=hide_element,
            start_fun='{',
            end_fun='}')
        return script

    def add_graph_to_report(self,
                            fig: 'plotly.graph_objs._figure.Figure',
                            title: str):
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
        print('Added graph to report!')

    def add_many_plots_with_selection_box_to_report(self,
                                                    figures_dict: dict,
                                                    title: str, **kwargs):
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
        labels = kwargs.get('labels', 'Choose a label')
        self.html_report += self.select_plotly(figures_dict, title, labels=labels)
        print('Added graph to report!')

    def open_section(self,
                     section_title: str, ingore_multi_section=False
                     ) -> 'html plotly code':
        if not self.section_just_open or ingore_multi_section:
            self.html_report += f'<button class="collapsiblemygs">{section_title}</button> <div class="content">'
            self.section_just_open = True
            print('Open section to report!')
        else:
            print('Warning you already have an open section')

    def close_section(self, ingore_multi_section=False) -> 'html plotly code':
        if self.section_just_open or ingore_multi_section:
            self.html_report += f'</div>'
            self.section_just_open = False
            print('Close section to report!')
        else:
            print('Warning you did not open section yet')

    def add_title(self, title: str) -> 'html plotly code':
        self.html_report += f'<h2>{title}</h2>'
        print('Added title to report!')

    def add_text(self, text: str) -> 'html plotly code':
        self.html_report += f'<p>{text}</p>'
        print('Added text to report!')

    def add_dataframe(self, df: pd.DataFrame, title: str, limit_max=2000) -> 'html plotly code':
        if len(df) < limit_max:
            self.html_report += f'<a href="#" onclick="download_table_as_csv('+f"'{title.replace(' ', '')}'"+');">Download as CSV</a>'
            tab = df.to_html().replace('class="dataframe"', f'id="{title.replace(" ", "")}"').replace('border="1"', 'border="0"')
            self.html_report += f'<div id="table-wrapper"><div id="table-scroll">{tab}</div></div>'
            print('Added table to report!')
        else:
            print('The DataFrame is to big!')

    def save_html(self, file_name='export.html'):
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
        with open(file_name, 'w', encoding="utf-8") as f:
            f.write(html_txt)
        print('Saved!')
