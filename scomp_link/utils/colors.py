# -*- coding: utf-8 -*-
"""
 ██████╗ ██████╗ ██╗      ██████╗ ██████╗ ███████╗
██╔════╝██╔═══██╗██║     ██╔═══██╗██╔══██╗██╔════╝
██║     ██║   ██║██║     ██║   ██║██████╔╝███████╗
██║     ██║   ██║██║     ██║   ██║██╔══██╗╚════██║
╚██████╗╚██████╔╝███████╗╚██████╔╝██║  ██║███████║
 ╚═════╝ ╚═════╝ ╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝

Centralized color palettes for all scomp-link visualizations.
Import colors from here — never hardcode hex values elsewhere.
"""

# Primary palette — 10 distinct colors for categorical data
PRIMARY = ["#6E37FA", "#32BBB9", "#FF9408", "#F40953", "#FA32A0",
           "#B30095", "#FFD500", "#AAF564", "#50E6AA", "#2765F0"]

# Lighter variants (backgrounds, hover states)
LIGHT = ["#EAE4FF", "#C4EDEC", "#FEDEB4", "#FFAEBB", "#FFB4DD",
         "#E694D9", "#FFF4B3", "#DCFFAE", "#C1F6DC", "#DCF0FF"]

# Medium-light variants
MEDIUM_LIGHT = ["#DCD2FF", "#88DCDA", "#FFBE6A", "#FF8296", "#FF8CC8",
                "#DB64C7", "#FFEA69", "#C8FF78", "#8FEFC8", "#AADCFA"]

# Medium variants
MEDIUM = ["#BEB4FF", "#009DA8", "#FA7D00", "#FF3C64", "#FF5AB4",
          "#D13CBD", "#FFC800", "#8CE650", "#00C896", "#64BEF0"]

# Medium-dark variants
MEDIUM_DARK = ["#9682FF", "#00889B", "#EB6E00", "#DC0046", "#E11E8C",
               "#C800AA", "#F0B300", "#74D22C", "#00AA8C", "#1E96F0"]

# Dark variants
DARK = ["#5528D2", "#007396", "#D75F00", "#BE003C", "#C80082",
        "#81006E", "#E19B00", "#46B900", "#008C78", "#0049BE"]

# Darkest variants
DARKEST = ["#4614B4", "#005F81", "#C35000", "#A00032", "#B40073",
           "#64005A", "#D29100", "#259600", "#007C6E", "#003CA0"]

# Theme colors
MAIN = "#6E37FA"
MAIN_LIGHT = "#9682FF"
MAIN_DARK = "#4614B4"

# JSON string for Highcharts embedding
PRIMARY_JSON = '["#6E37FA", "#32BBB9", "#FF9408", "#F40953", "#FA32A0", "#B30095", "#FFD500", "#AAF564", "#50E6AA", "#2765F0"]'
