# -*- coding: utf-8 -*-
"""
Example 29: Validator + AdvancedCV — Detailed Usage
====================================================

Demonstrates standalone use of validation components:
  1. Validator.evaluate() for regression and classification metrics
  2. Validator.k_fold_cv() — K-Fold cross-validation
  3. Validator.loocv() — Leave-One-Out cross-validation (small dataset)
  4. Validator.generate_validation_report() — HTML report generation
  5. AdvancedCV.evaluate_all() — Bootstrap + K-Fold + optional LOOCV

Requirements:
  pip install scomp-link
"""

import os
import tempfile
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split

from scomp_link import Validator
from scomp_link.validation.advanced_cv import AdvancedCV
from scomp_link.utils.decorators import timer, memory_usage


# --- Helper functions with decorators ---

@memory_usage
def prepare_regression_data(n: int = 500):
    """Generate regression data, train model, return all components."""
    np.random.seed(42)
    X = pd.DataFrame({
        'feature_a': np.random.randn(n),
        'feature_b': np.random.randn(n) * 2,
        'feature_c': np.random.exponential(1, n),
        'feature_d': np.random.uniform(-3, 3, n),
    })
    y = 3 * X['feature_a'] - 2 * X['feature_b'] + 0.5 * X['feature_c'] + np.random.randn(n) * 0.5

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = GradientBoostingRegressor(n_estimators=80, max_depth=4, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    return model, X, y, X_train, X_test, y_train, y_test, y_pred


@timer
def run_advanced_cv(model, X, y, include_loocv: bool = False):
    """Run AdvancedCV.evaluate_all with configurable methods."""
    results = AdvancedCV.evaluate_all(
        model, X, y,
        include_loocv=include_loocv,
        include_bootstrap=True,
        bootstrap_iterations=200,
    )
    return results


# --- Main execution ---

if __name__ == '__main__':
    print("=" * 70)
    print("VALIDATOR + ADVANCED CV — DETAILED USAGE")
    print("=" * 70)

    # ═══════════════════════════════════════════════════════
    # PART 1: REGRESSION VALIDATION
    # ═══════════════════════════════════════════════════════

    print("\n" + "═" * 50)
    print("PART 1: REGRESSION VALIDATION")
    print("═" * 50)

    # === 1. Prepare model and data ===
    print("\n--- 1. Preparing regression model ---")
    model, X, y, X_train, X_test, y_train, y_test, y_pred = prepare_regression_data(500)
    print(f"  Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    print(f"  Model: {type(model).__name__}")
    print(f"  R² on test: {model.score(X_test, y_test):.4f}")

    # === 2. Validator.evaluate ===
    print("\n--- 2. Validator.evaluate() — Regression Metrics ---")
    validator = Validator(model)
    metrics = validator.evaluate(y_test, y_pred, task_type="regression")
    print(f"  MSE:  {metrics['mse']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  R²:   {metrics['r2']:.4f}")

    # === 3. K-Fold CV ===
    print("\n--- 3. Validator.k_fold_cv() — 5-Fold ---")
    cv_scores = validator.k_fold_cv(X, y, k=5)
    print(f"  Fold scores: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"  Mean: {np.mean(cv_scores):.4f} (±{np.std(cv_scores):.4f})")

    # === 4. LOOCV (small subset for speed) ===
    print("\n--- 4. Validator.loocv() — Leave-One-Out (subset of 80 samples) ---")
    X_small = X.iloc[:80]
    y_small = y.iloc[:80]
    loocv_scores = validator.loocv(X_small, y_small)
    print(f"  LOOCV Mean Score: {np.mean(loocv_scores):.4f}")
    print(f"  LOOCV Std:        {np.std(loocv_scores):.4f}")
    print(f"  Num evaluations:  {len(loocv_scores)}")

    # === 5. Generate HTML validation report ===
    print("\n--- 5. Generating HTML validation report ---")
    report_path = tempfile.mktemp(suffix='.html')
    validator.generate_validation_report(
        y_test, y_pred,
        task_type="regression",
        report_name=report_path,
    )
    size_kb = os.path.getsize(report_path) / 1024
    print(f"  ✅ Report saved: {size_kb:.1f} KB")
    os.unlink(report_path)
    print(f"  (cleaned up temp file)")

    # === 6. AdvancedCV — Bootstrap + K-Fold ===
    print("\n--- 6. AdvancedCV.evaluate_all() — Bootstrap + K-Fold ---")
    advanced_results = run_advanced_cv(model, X, y, include_loocv=False)
    print(f"\n  Results:")
    for key, result in advanced_results.items():
        print(f"    {result['method']}: mean={result['mean_score']:.4f} (±{result['std_score']:.4f})")
        if 'ci_lower' in result:
            print(f"      95% CI: [{result['ci_lower']:.4f}, {result['ci_upper']:.4f}]")

    # ═══════════════════════════════════════════════════════
    # PART 2: CLASSIFICATION VALIDATION
    # ═══════════════════════════════════════════════════════

    print("\n\n" + "═" * 50)
    print("PART 2: CLASSIFICATION VALIDATION")
    print("═" * 50)

    # === 7. Classification setup ===
    print("\n--- 7. Preparing classification model ---")
    np.random.seed(42)
    n_cls = 400
    X_cls = pd.DataFrame({
        'f1': np.random.randn(n_cls),
        'f2': np.random.randn(n_cls),
        'f3': np.random.randn(n_cls),
    })
    y_cls = ((X_cls['f1'] + 0.5 * X_cls['f2']) > 0).astype(int)

    X_tr, X_te, y_tr, y_te = train_test_split(X_cls, y_cls, test_size=0.25, random_state=42)
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_tr, y_tr)
    y_pred_cls = clf.predict(X_te)
    y_proba_cls = clf.predict_proba(X_te)

    print(f"  Accuracy: {(y_pred_cls == y_te).mean():.4f}")

    # === 8. Validator.evaluate — Classification ===
    print("\n--- 8. Validator.evaluate() — Classification Metrics ---")
    val_cls = Validator(clf)
    cls_metrics = val_cls.evaluate(y_te, y_pred_cls, task_type="classification", y_proba=y_proba_cls)
    print(f"  Accuracy:  {cls_metrics['accuracy']:.4f}")
    print(f"  F1:        {cls_metrics['f1']:.4f}")
    print(f"  Precision: {cls_metrics['precision']:.4f}")
    print(f"  Recall:    {cls_metrics['recall']:.4f}")

    # === 9. Classification validation report ===
    print("\n--- 9. Classification validation report ---")
    report_path_cls = tempfile.mktemp(suffix='.html')
    val_cls.generate_validation_report(
        y_te, y_pred_cls,
        task_type="classification",
        y_proba=y_proba_cls,
        report_name=report_path_cls,
    )
    size_kb_cls = os.path.getsize(report_path_cls) / 1024
    print(f"  ✅ Classification report saved: {size_kb_cls:.1f} KB")
    os.unlink(report_path_cls)

    # === 10. AdvancedCV on classifier ===
    print("\n--- 10. AdvancedCV on classifier (K-Fold + Bootstrap) ---")
    adv_cls = AdvancedCV.evaluate_all(
        clf, X_cls, y_cls,
        include_loocv=False,
        include_bootstrap=True,
        bootstrap_iterations=150,
    )
    for key, result in adv_cls.items():
        print(f"  {result['method']}: mean={result['mean_score']:.4f} (±{result['std_score']:.4f})")

    # === 11. PDF Conversion ===
    print("\n--- 11. PDF Conversion ---")
    from scomp_link.utils.pdf_converter import HAS_WEASYPRINT, HAS_MARKDOWN, _wrap_html

    # Exercise _wrap_html regardless of weasyprint availability
    wrapped = _wrap_html("<h1>Test</h1><p>Coverage test content</p>")
    print(f"  _wrap_html output length: {len(wrapped)} chars")

    wrapped_custom_css = _wrap_html("<p>Custom styled</p>", css="body { color: red; }")
    print(f"  _wrap_html with custom CSS: {len(wrapped_custom_css)} chars")

    if HAS_WEASYPRINT:
        from scomp_link.utils.pdf_converter import html_to_pdf, markdown_to_pdf

        # Generate a fresh HTML report for PDF conversion
        html_path = tempfile.mktemp(suffix='.html')
        validator.generate_validation_report(y_test, y_pred, task_type="regression", report_name=html_path)

        # Convert HTML → PDF
        pdf_path = tempfile.mktemp(suffix='.pdf')
        result_path = html_to_pdf(html_path, output_path=pdf_path)
        pdf_size = os.path.getsize(result_path) / 1024
        print(f"  ✅ HTML→PDF (WeasyPrint): {pdf_size:.1f} KB")
        os.unlink(html_path)
        os.unlink(pdf_path)

        # Convert Markdown → PDF
        md_path = tempfile.mktemp(suffix='.md')
        with open(md_path, 'w') as f:
            f.write("# Validation Summary\n\n")
            f.write("| Metric | Value |\n|--------|-------|\n")
            for k, v in metrics.items():
                f.write(f"| {k} | {v:.4f} |\n")
            f.write("\n## Conclusion\n\nModel performance is acceptable.\n")

        pdf_md_path = tempfile.mktemp(suffix='.pdf')
        result_md = markdown_to_pdf(md_path, output_path=pdf_md_path)
        pdf_md_size = os.path.getsize(result_md) / 1024
        print(f"  ✅ Markdown→PDF (WeasyPrint): {pdf_md_size:.1f} KB")
        os.unlink(md_path)
        os.unlink(pdf_md_path)
    else:
        print("  ⚠️  WeasyPrint not available, testing Playwright PDF path...")
        # Test that html_to_pdf / markdown_to_pdf raise ImportError properly
        from scomp_link.utils.pdf_converter import html_to_pdf, markdown_to_pdf
        try:
            html_to_pdf("/tmp/nonexistent.html")
        except ImportError as e:
            print(f"  ✅ html_to_pdf raises ImportError correctly: {str(e)[:50]}")

        try:
            markdown_to_pdf("/tmp/nonexistent.md")
        except ImportError as e:
            print(f"  ✅ markdown_to_pdf raises ImportError correctly: {str(e)[:50]}")

        # Fallback: Playwright-based PDF via ScompLinkHTMLReport
        from scomp_link.utils.report_html import ScompLinkHTMLReport
        report_pdf = ScompLinkHTMLReport(title='PDF Test Report')
        report_pdf.add_title("Validation Metrics")
        import pandas as pd_local
        metrics_df = pd_local.DataFrame([metrics])
        report_pdf.add_dataframe(metrics_df, "Metrics Table")

        html_path = tempfile.mktemp(suffix='.html')
        report_pdf.save_html(html_path)
        pdf_path = html_path.replace('.html', '.pdf')
        try:
            report_pdf.save_pdf(pdf_path)
            pdf_size = os.path.getsize(pdf_path) / 1024
            print(f"  ✅ Playwright PDF: {pdf_size:.1f} KB")
            os.unlink(pdf_path)
        except Exception as e:
            print(f"  ⚠️  Playwright PDF skipped: {type(e).__name__}")
        os.unlink(html_path)

    print(f"  HAS_WEASYPRINT={HAS_WEASYPRINT}, HAS_MARKDOWN={HAS_MARKDOWN}")

    # === Summary ===
    print("\n" + "=" * 70)
    print("✅ Validator + AdvancedCV example complete!")
    print("   • Regression: evaluate, k_fold_cv, loocv, HTML report")
    print("   • Classification: evaluate, HTML report with probabilities")
    print("   • AdvancedCV: K-Fold + Bootstrap with confidence intervals")
    print("   • PDF conversion: attempted (weasyprint or playwright)")
    print("=" * 70)
