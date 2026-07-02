# -*- coding: utf-8 -*-
"""
Example 32: Highcharts Visualizations in HTML Report
=====================================================

Demonstrates all Highcharts chart types embedded in a ScompLinkHTMLReport:
  1. Streamgraph (area mode) — multi-series time evolution
  2. Streamgraph (streamgraph mode) — symmetric flow visualization
  3. Calendar Heatmap — daily percentage values
  4. Calendar Gantt — project timeline with milestones

All charts are generated with synthetic data and saved into a single
self-contained HTML report file.

Requirements:
  pip install scomp-link
"""

import os
import tempfile
import numpy as np
from datetime import datetime, timedelta

from scomp_link.utils.report_html import ScompLinkHTMLReport
from scomp_link.utils.highcharts import streamgraphs, calendar_heatmap, calendar_gantt
from scomp_link.utils.decorators import timer, memory_usage


# --- Data generation functions with decorators ---

@memory_usage
def generate_streamgraph_data():
    """Generate multi-series time data for streamgraphs."""
    np.random.seed(42)

    # Monthly dates for 2 years
    start = datetime(2024, 1, 1)
    dates = [(start + timedelta(days=30 * i)).strftime('%Y-%m') for i in range(24)]

    # 5 product categories with seasonal patterns
    categories = {
        'Electronics': np.random.poisson(50, 24) + np.array([10 * (i % 12 > 9) for i in range(24)]),
        'Clothing': np.random.poisson(40, 24) + np.array([15 * (3 < i % 12 < 8) for i in range(24)]),
        'Food': np.random.poisson(80, 24) + np.random.randint(-5, 5, 24),
        'Books': np.random.poisson(25, 24) + np.array([8 * (i % 12 < 3) for i in range(24)]),
        'Sports': np.random.poisson(30, 24) + np.array([20 * (4 < i % 12 < 9) for i in range(24)]),
    }

    # Convert to list of ints (as required by the function)
    series_dict = {k: [int(v) for v in vals] for k, vals in categories.items()}

    # Annotations for key events
    annotations = {
        'Black Friday': 10,   # November 2024
        'Summer Sale': 18,    # July 2025
    }

    return dates, series_dict, annotations


@memory_usage
def generate_heatmap_data():
    """Generate daily percentage data for calendar heatmap."""
    np.random.seed(123)

    # 35 days of data (5 weeks)
    start = datetime(2025, 3, 1)
    series_dict = {}
    for i in range(35):
        date = (start + timedelta(days=i)).strftime('%Y-%m-%d')
        # Simulate daily completion percentage with weekday bias
        weekday = (start + timedelta(days=i)).weekday()
        base = 60 if weekday < 5 else 30  # lower on weekends
        value = round(min(100, max(0, base + np.random.randn() * 20)), 2)
        series_dict[date] = value

    return series_dict


@memory_usage
def generate_gantt_data():
    """Generate project management Gantt chart data."""
    # Project phases with start/end timestamps (milliseconds since epoch)
    base = datetime(2025, 6, 1)

    def to_ms(dt):
        return f"Date.UTC({dt.year}, {dt.month - 1}, {dt.day})"

    series_dict = [
        {
            'name': 'Development',
            'data': [
                {
                    'name': 'Requirements',
                    'id': 'req',
                    'start': to_ms(base),
                    'end': to_ms(base + timedelta(days=7)),
                    'completed': "{ amount: 1.0 }",
                },
                {
                    'name': 'Design',
                    'id': 'design',
                    'start': to_ms(base + timedelta(days=7)),
                    'end': to_ms(base + timedelta(days=14)),
                    'completed': "{ amount: 0.8 }",
                },
                {
                    'name': 'Implementation',
                    'id': 'impl',
                    'start': to_ms(base + timedelta(days=14)),
                    'end': to_ms(base + timedelta(days=35)),
                    'completed': "{ amount: 0.4 }",
                },
                {
                    'name': 'Code Freeze',
                    'id': 'freeze',
                    'start': to_ms(base + timedelta(days=35)),
                    'end': to_ms(base + timedelta(days=35)),
                    'milestone': 'true',
                },
            ],
        },
        {
            'name': 'Testing',
            'data': [
                {
                    'name': 'Unit Tests',
                    'id': 'unit',
                    'start': to_ms(base + timedelta(days=21)),
                    'end': to_ms(base + timedelta(days=35)),
                    'completed': "{ amount: 0.6 }",
                },
                {
                    'name': 'Integration Tests',
                    'id': 'integ',
                    'start': to_ms(base + timedelta(days=35)),
                    'end': to_ms(base + timedelta(days=42)),
                    'completed': "{ amount: 0.2 }",
                },
                {
                    'name': 'Release',
                    'id': 'release',
                    'start': to_ms(base + timedelta(days=45)),
                    'end': to_ms(base + timedelta(days=45)),
                    'milestone': 'true',
                },
            ],
        },
    ]

    min_date = base.strftime('%Y-%m-%d')
    max_date = (base + timedelta(days=50)).strftime('%Y-%m-%d')

    return series_dict, min_date, max_date


@timer
def build_report():
    """Build the complete HTML report with all Highcharts visualizations."""
    report = ScompLinkHTMLReport(title='Highcharts Visualization Gallery')

    # === Section 1: Streamgraph (Area mode) ===
    print("\n  📊 Generating Streamgraph (Area mode)...")
    dates, series_dict, annotations = generate_streamgraph_data()

    report.open_section("Streamgraph — Area Mode")
    report.add_text("Multi-series area chart showing product category sales over 24 months.")
    html_area = streamgraphs(
        title="Monthly Sales by Category (Area)",
        dates=dates,
        series_dict=series_dict,
        annotation=annotations,
        area=True,
    )
    report.html_report += html_area
    report.close_section()
    print(f"    ✅ Area chart: {len(series_dict)} series, {len(dates)} time points")

    # === Section 2: Streamgraph (Streamgraph mode) ===
    print("  📊 Generating Streamgraph (Stream mode)...")
    report.open_section("Streamgraph — Symmetric Stream Mode")
    report.add_text("Same data as above but rendered as a symmetric streamgraph.")
    html_stream = streamgraphs(
        title="Monthly Sales by Category (Stream)",
        dates=dates,
        series_dict=series_dict,
        annotation=None,
        area=False,
    )
    report.html_report += html_stream
    report.close_section()
    print(f"    ✅ Streamgraph: symmetric layout")

    # === Section 3: Calendar Heatmap ===
    print("  📊 Generating Calendar Heatmap...")
    heatmap_data = generate_heatmap_data()

    report.open_section("Calendar Heatmap — Daily Completion Rate")
    report.add_text("Daily task completion percentage over 5 weeks. "
                    "Color scale: blue (low) → orange (high).")
    html_heatmap = calendar_heatmap(
        title="Daily Task Completion (%)",
        series_dict=heatmap_data,
        min=0,
        max=100,
    )
    report.html_report += html_heatmap
    report.close_section()
    print(f"    ✅ Heatmap: {len(heatmap_data)} days")

    # === Section 4: Gantt Chart ===
    print("  📊 Generating Gantt Chart...")
    gantt_data, min_date, max_date = generate_gantt_data()

    report.open_section("Gantt Chart — Project Timeline")
    report.add_text("Project management timeline with phases, tasks, and milestones. "
                    "Includes completion percentages and weekend shading.")
    html_gantt = calendar_gantt(
        title="Project Alpha — Development Timeline",
        series_dict=gantt_data,
        min_date=min_date,
        max_date=max_date,
    )
    report.html_report += html_gantt
    report.close_section()
    print(f"    ✅ Gantt: {sum(len(s['data']) for s in gantt_data)} tasks, "
          f"range {min_date} → {max_date}")

    return report


# --- Main execution ---

if __name__ == '__main__':
    print("=" * 70)
    print("HIGHCHARTS VISUALIZATIONS IN HTML REPORT")
    print("=" * 70)

    # Build the full report
    report = build_report()

    # Save to temp file
    print("\n--- Saving HTML Report ---")
    output_path = tempfile.mktemp(suffix='.html')
    report.save_html(output_path)

    file_size = os.path.getsize(output_path) / 1024
    print(f"  📁 Report saved: {output_path}")
    print(f"  📏 File size: {file_size:.1f} KB")

    # Verify content includes all chart types
    with open(output_path, 'r', encoding='utf-8') as f:
        content = f.read()

    checks = {
        'Highcharts.chart': 'Highcharts core charts',
        'HighchartsGantt.ganttChart': 'Gantt chart',
        'type: \'heatmap\'': 'Heatmap chart',
        'streamgraph': 'Streamgraph reference',
    }

    print(f"\n  --- Content Verification ---")
    for pattern, label in checks.items():
        found = pattern in content
        status = "✅" if found else "❌"
        print(f"    {status} {label}: {'found' if found else 'MISSING'}")

    # Cleanup
    os.unlink(output_path)
    print(f"\n  (temp file cleaned up)")

    # === Summary ===
    print("\n" + "=" * 70)
    print("✅ Highcharts report example complete!")
    print("   • Streamgraph (area): multi-series with annotations")
    print("   • Streamgraph (stream): symmetric layout")
    print("   • Calendar Heatmap: daily percentages with color scale")
    print("   • Gantt Chart: project timeline with milestones")
    print(f"   • Total report size: {file_size:.1f} KB")
    print("=" * 70)
