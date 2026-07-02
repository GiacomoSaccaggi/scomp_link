# -*- coding: utf-8 -*-
"""
Example 31: CLI Programmatic Usage
====================================

Demonstrates programmatic invocation of all scomp-link CLI commands:
  1. run — train a model and save artifact
  2. predict — load artifact and predict on new data
  3. quality — data quality report
  4. engineer — feature engineering
  5. drift — detect distribution drift
  6. anomaly — anomaly detection
  7. fairness — fairness metrics
  8. forecast — time series forecasting
  9. compare — compare two artifacts
  10. info — inspect an artifact
  11. init — scaffold a new project
  12. report — generate HTML report

Requirements:
  pip install scomp-link
"""

import os
import sys
import shutil
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path

from scomp_link.cli import build_parser
from scomp_link.utils.decorators import timer, suppress_exceptions


# --- Helper functions ---

@timer
def create_test_datasets(tmp_dir: Path):
    """Create CSV files for testing all CLI commands."""
    np.random.seed(42)
    n = 200

    # Regression dataset
    df_reg = pd.DataFrame({
        'x1': np.random.randn(n),
        'x2': np.random.randn(n) * 2,
        'x3': np.random.uniform(-1, 1, n),
        'target': 3 * np.random.randn(n) + 1,
    })
    df_reg['target'] = 2 * df_reg['x1'] - df_reg['x2'] + 0.5 * df_reg['x3'] + np.random.randn(n) * 0.3
    path_reg = tmp_dir / "regression.csv"
    df_reg.to_csv(path_reg, index=False)

    # Classification dataset (for fairness)
    df_cls = pd.DataFrame({
        'feature_a': np.random.randn(n),
        'feature_b': np.random.randn(n),
        'y_true': np.random.choice([0, 1], n),
        'y_pred': np.random.choice([0, 1], n),
        'gender': np.random.choice(['male', 'female'], n),
    })
    path_cls = tmp_dir / "classification.csv"
    df_cls.to_csv(path_cls, index=False)

    # Time series dataset
    t = np.arange(n)
    df_ts = pd.DataFrame({
        'value': 50 + 0.3 * t + 10 * np.sin(2 * np.pi * t / 30) + np.random.randn(n) * 3,
    })
    path_ts = tmp_dir / "timeseries.csv"
    df_ts.to_csv(path_ts, index=False)

    # Drift reference and current (shifted)
    df_ref = pd.DataFrame({
        'a': np.random.normal(0, 1, n),
        'b': np.random.normal(5, 2, n),
        'c': np.random.uniform(0, 10, n),
    })
    df_current = pd.DataFrame({
        'a': np.random.normal(2, 1, n),  # shifted mean
        'b': np.random.normal(5, 2, n),  # same
        'c': np.random.uniform(0, 10, n),  # same
    })
    path_ref = tmp_dir / "reference.csv"
    path_current = tmp_dir / "current.csv"
    df_ref.to_csv(path_ref, index=False)
    df_current.to_csv(path_current, index=False)

    return {
        'regression': str(path_reg),
        'classification': str(path_cls),
        'timeseries': str(path_ts),
        'reference': str(path_ref),
        'current': str(path_current),
    }


@suppress_exceptions(default="❌ FAILED", log=True)
def run_cli_command(description: str, args_list: list):
    """Parse and execute a CLI command, returning status."""
    parser = build_parser()
    args = parser.parse_args(args_list)
    args.func(args)
    return "✅ OK"


# --- Main execution ---

if __name__ == '__main__':
    print("=" * 70)
    print("CLI PROGRAMMATIC USAGE — ALL COMMANDS")
    print("=" * 70)

    tmp_dir = Path(tempfile.mkdtemp())
    results = {}

    try:
        # === Create test data ===
        print("\n--- Creating test datasets ---")
        datasets = create_test_datasets(tmp_dir)
        print(f"  Created {len(datasets)} CSV files in temp directory")

        artifact_path = str(tmp_dir / "model.scomp")
        artifact_path_2 = str(tmp_dir / "model_2.scomp")

        # === 1. run ===
        print("\n--- 1. scomp-link run ---")
        results['run'] = run_cli_command("run", [
            'run',
            '--data', datasets['regression'],
            '--target', 'target',
            '--task', 'regression',
            '--save-artifact', artifact_path,
            '--silent',
        ])
        print(f"  {results['run']}")

        # === 2. predict ===
        print("\n--- 2. scomp-link predict ---")
        predictions_path = str(tmp_dir / "predictions.csv")
        results['predict'] = run_cli_command("predict", [
            'predict',
            '--artifact', artifact_path,
            '--data', datasets['regression'],
            '--output', predictions_path,
            '--silent',
        ])
        print(f"  {results['predict']}")

        # === 3. info ===
        print("\n--- 3. scomp-link info ---")
        results['info'] = run_cli_command("info", [
            'info',
            '--artifact', artifact_path,
        ])
        print(f"  {results['info']}")

        # === 4. quality ===
        print("\n--- 4. scomp-link quality ---")
        quality_path = str(tmp_dir / "quality.html")
        results['quality'] = run_cli_command("quality", [
            'quality',
            '--data', datasets['regression'],
            '--output', quality_path,
            '--silent',
        ])
        print(f"  {results['quality']}")

        # === 5. engineer ===
        print("\n--- 5. scomp-link engineer ---")
        eng_path = str(tmp_dir / "engineered.csv")
        results['engineer'] = run_cli_command("engineer", [
            'engineer',
            '--data', datasets['regression'],
            '--target', 'target',
            '--output', eng_path,
            '--interactions',
            '--log-transform',
            '--silent',
        ])
        print(f"  {results['engineer']}")

        # === 6. drift ===
        print("\n--- 6. scomp-link drift ---")
        results['drift'] = run_cli_command("drift", [
            'drift',
            '--reference', datasets['reference'],
            '--current', datasets['current'],
            '--silent',
        ])
        print(f"  {results['drift']}")

        # === 7. anomaly ===
        print("\n--- 7. scomp-link anomaly ---")
        anomaly_path = str(tmp_dir / "anomalies.csv")
        results['anomaly'] = run_cli_command("anomaly", [
            'anomaly',
            '--data', datasets['reference'],
            '--methods', 'iforest,lof',
            '--output', anomaly_path,
            '--silent',
        ])
        print(f"  {results['anomaly']}")

        # === 8. fairness ===
        print("\n--- 8. scomp-link fairness ---")
        results['fairness'] = run_cli_command("fairness", [
            'fairness',
            '--data', datasets['classification'],
            '--target', 'y_true',
            '--predicted', 'y_pred',
            '--sensitive', 'gender',
            '--silent',
        ])
        print(f"  {results['fairness']}")

        # === 9. forecast ===
        print("\n--- 9. scomp-link forecast ---")
        forecast_path = str(tmp_dir / "forecast.csv")
        results['forecast'] = run_cli_command("forecast", [
            'forecast',
            '--data', datasets['timeseries'],
            '--column', 'value',
            '--horizon', '10',
            '--method', 'arima',
            '--output', forecast_path,
            '--silent',
        ])
        print(f"  {results['forecast']}")

        # === 10. compare (need 2 artifacts) ===
        print("\n--- 10. scomp-link compare ---")
        # Create second artifact
        run_cli_command("run_2", [
            'run',
            '--data', datasets['regression'],
            '--target', 'target',
            '--task', 'regression',
            '--save-artifact', artifact_path_2,
            '--test-size', '0.3',
            '--silent',
        ])
        compare_path = str(tmp_dir / "comparison.csv")
        results['compare'] = run_cli_command("compare", [
            'compare',
            '--artifacts', artifact_path, artifact_path_2,
            '--output', compare_path,
        ])
        print(f"  {results['compare']}")

        # === 11. init ===
        print("\n--- 11. scomp-link init ---")
        init_path = str(tmp_dir / "my_project")
        results['init'] = run_cli_command("init", [
            'init', 'my_project',
            '--force',
        ])
        # Clean up the scaffolded project in cwd
        if Path("my_project").exists():
            shutil.rmtree("my_project")
        print(f"  {results['init']}")

        # === 12. report (EDA) ===
        print("\n--- 12. scomp-link report ---")
        report_path = str(tmp_dir / "eda_report.html")
        results['report'] = run_cli_command("report", [
            'report',
            '--data', datasets['regression'],
            '--output', report_path,
            '--silent',
        ])
        print(f"  {results['report']}")

        # === Summary ===
        print("\n" + "=" * 70)
        print("SUMMARY — CLI COMMANDS EXERCISED")
        print("=" * 70)
        success_count = sum(1 for v in results.values() if "OK" in str(v))
        total = len(results)
        print(f"\n  {success_count}/{total} commands executed successfully")
        print()
        for cmd, status in results.items():
            print(f"  {status}  {cmd}")

        # List generated files
        print(f"\n  Generated files in temp dir:")
        for f in sorted(tmp_dir.iterdir()):
            if f.is_file():
                size = f.stat().st_size / 1024
                print(f"    {f.name:<30} {size:.1f} KB")

    finally:
        # Cleanup
        shutil.rmtree(tmp_dir, ignore_errors=True)
        # Also clean up any scaffolded project
        if Path("my_project").exists():
            shutil.rmtree("my_project", ignore_errors=True)

    print("\n✅ CLI programmatic usage example complete!")
