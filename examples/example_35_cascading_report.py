# -*- coding: utf-8 -*-
"""
Example 35: Cascading Dropdowns, Styled Tables & Chart Utilities
================================================================

Demonstrates the new interactive report features:
1. add_cascading_content — multi-dimensional dropdown navigation
2. add_dataframe with thresholds — conditional formatting
3. index_chart — normalized index charts with toggle buttons
4. stacked_area_comparison — side-by-side 100% stacked areas
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from scomp_link.utils.plotly_utils import (
    fill_timeslots,
    index_chart,
    normalize_to_index,
    stacked_area_comparison,
)
from scomp_link.utils.report_html import ScompLinkHTMLReport


def main():
    # --- Setup report ---
    report = ScompLinkHTMLReport("Cascading Report Demo")
    report.add_title("Interactive Report Features Demo")
    report.add_text("This report demonstrates cascading dropdowns, styled tables, and chart utilities.")

    # ==========================================================
    # 1. Cascading Content with Plotly Figures
    # ==========================================================
    report.open_section("Cascading Dropdowns — Plotly Figures")
    report.add_text(
        "Select a region and metric to view the corresponding chart. "
        "Dropdowns update the visible content instantly without page reload."
    )

    # Generate sample figures for each combination
    np.random.seed(42)
    regions = ["North", "South", "East"]
    metrics = ["Revenue", "Users"]
    content_map = {}
    for region in regions:
        for metric in metrics:
            x = list(range(1, 13))
            y = np.random.randint(50, 200, size=12).tolist()
            fig = go.Figure(data=[go.Bar(x=x, y=y, marker_color="#6E37FA")])
            fig.update_layout(
                title=f"{region} — {metric}",
                xaxis_title="Month",
                yaxis_title=metric,
                height=350,
            )
            content_map[(region, metric)] = fig

    dimensions = [
        {"label": "Region", "options": regions},
        {"label": "Metric", "options": metrics},
    ]
    report.add_cascading_content("Regional Performance", dimensions, content_map)
    report.close_section()

    # ==========================================================
    # 2. Cascading Content with HTML (cascade=True)
    # ==========================================================
    report.open_section("Cascading Dropdowns — HTML Content (cascade mode)")
    report.add_text(
        "In cascade mode, the second dropdown options are filtered based on the first selection."
    )

    countries = ["US", "Germany"]
    cities_map = {"US": ["New York", "Los Angeles"], "Germany": ["Berlin", "Munich"]}
    content_cascade = {}
    for country, cities in cities_map.items():
        for city in cities:
            content_cascade[(country, city)] = (
                f"<div style='padding:1rem;background:#f0f4ff;border-radius:8px;'>"
                f"<h3>{city}, {country}</h3>"
                f"<p>Population data and statistics for {city} would appear here.</p>"
                f"</div>"
            )

    all_cities = [c for cities in cities_map.values() for c in cities]
    dims_cascade = [
        {"label": "Country", "options": countries},
        {"label": "City", "options": all_cities},
    ]
    report.add_cascading_content("City Explorer", dims_cascade, content_cascade, cascade=True)
    report.close_section()

    # ==========================================================
    # 3. Styled DataFrame with Thresholds
    # ==========================================================
    report.open_section("Styled Tables with Conditional Formatting")
    report.add_text(
        "Tables with threshold-based coloring: green = good, orange = warning, red = critical."
    )

    metrics_df = pd.DataFrame({
        "model": ["RandomForest", "XGBoost", "LinearReg", "SVM", "KNN"],
        "rmse": [0.08, 0.12, 0.28, 0.19, 0.35],
        "r2_score": [0.95, 0.88, 0.72, 0.55, 0.40],
        "train_time_s": [2.1, 1.8, 0.3, 5.2, 0.8],
    })

    thresholds = {
        "rmse": (0.10, 0.25, False),       # lower is better
        "r2_score": (0.85, 0.60, True),    # higher is better
        "train_time_s": (2.0, 4.0, False), # lower is better
    }

    report.add_dataframe(metrics_df, "Model Comparison", thresholds=thresholds)
    report.close_section()

    # ==========================================================
    # 4. Index Chart
    # ==========================================================
    report.open_section("Index Chart Utility")
    report.add_text(
        "Index charts normalize series to baseline=100 (daily average). "
        "Use toggle buttons to switch between groups."
    )

    hours = [f"{h:02d}:00" for h in range(24)]
    np.random.seed(123)

    fig_index = index_chart(
        series_dict={
            "Young (18-34)": {
                "solid": np.random.exponential(3, 24).tolist(),
                "dashed": (np.random.exponential(3, 24) * 0.9).tolist(),
            },
            "Middle (35-54)": {
                "solid": (np.random.normal(5, 1, 24).clip(0)).tolist(),
                "dashed": (np.random.normal(5, 1.2, 24).clip(0)).tolist(),
            },
            "Senior (55+)": {
                "solid": (np.random.normal(4, 0.8, 24).clip(0)).tolist(),
                "dashed": (np.random.normal(4, 1, 24).clip(0)).tolist(),
            },
        },
        x_labels=hours,
        title="Viewing Index by Age Group",
        baseline=100,
    )
    report.add_graph_to_report(fig_index, "Viewing Index")
    report.close_section()

    # ==========================================================
    # 5. Stacked Area Comparison
    # ==========================================================
    report.open_section("Stacked Area Comparison Utility")
    report.add_text(
        "Side-by-side stacked areas show composition differences between two sources. "
        "Each slot sums to 100%."
    )

    time_labels = [f"{h:02d}:{'00' if h % 2 == 0 else '30'}" for h in range(12)]
    fig_stacked = stacked_area_comparison(
        data_left={"Youth": [30, 25, 20, 15, 10, 15, 20, 30, 35, 30, 25, 20],
                   "Adults": [50, 55, 60, 65, 70, 65, 60, 50, 45, 50, 55, 60],
                   "Seniors": [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]},
        data_right={"Youth": [25, 22, 18, 12, 8, 12, 18, 28, 32, 28, 22, 18],
                    "Adults": [55, 58, 62, 68, 72, 68, 62, 52, 48, 52, 58, 62],
                    "Seniors": [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]},
        categories=["Youth", "Adults", "Seniors"],
        x_labels=time_labels,
        title="Audience Composition: Source A vs Source B",
        subplot_titles=("Source A", "Source B"),
    )
    report.add_graph_to_report(fig_stacked, "Composition Comparison")
    report.close_section()

    # ==========================================================
    # 6. Utility functions demo
    # ==========================================================
    report.open_section("Utility Functions")

    # fill_timeslots
    partial_data = [10, 20, 30]
    filled = fill_timeslots(partial_data, n_slots=8, fill_value=0)
    report.add_text(f"<code>fill_timeslots([10, 20, 30], n_slots=8)</code> → {filled.tolist()}")

    # normalize_to_index
    raw = [2, 4, 6, 8]
    indexed = normalize_to_index(raw, baseline=100)
    report.add_text(f"<code>normalize_to_index([2, 4, 6, 8], baseline=100)</code> → {[round(v, 1) for v in indexed.tolist()]}")

    report.close_section()

    # --- Save ---
    output_path = "cascading_report_demo.html"
    report.save_html(output_path)
    print(f"Report saved: {output_path}")


if __name__ == "__main__":
    main()
