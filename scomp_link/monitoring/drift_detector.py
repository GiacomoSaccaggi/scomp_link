# -*- coding: utf-8 -*-
"""
██████╗ ██████╗ ██╗███████╗████████╗
██╔══██╗██╔══██╗██║██╔════╝╚══██╔══╝
██║  ██║██████╔╝██║█████╗     ██║   
██║  ██║██╔══██╗██║██╔══╝     ██║   
██████╔╝██║  ██║██║██║        ██║   
╚═════╝ ╚═╝  ╚═╝╚═╝╚═╝        ╚═╝   
"""
import numpy as np
import pandas as pd
from typing import Optional, List, Dict
from scipy import stats

from scomp_link.utils.logger import get_logger
logger = get_logger(__name__)
from scomp_link.utils.decorators import timer, validate_args



class DriftDetector:
    """
    Detect data drift between a reference (training) dataset and a production dataset.
    Uses KS test for numerical features and PSI for all features.
    
    Dependencies: scipy, numpy, pandas, plotly
    
    PARAMETERS:
     1. reference: reference DataFrame (training data)
     2. psi_bins: number of bins for PSI calculation (default 10)
     3. ks_alpha: significance level for KS test (default 0.05)
     4. psi_threshold: PSI > threshold means significant drift (default 0.2)
    
    Usage example:
        detector = DriftDetector(X_train)
        report = detector.detect(X_production)
        fig = detector.plot_drift_report(report)
    """

    def __init__(self, reference: pd.DataFrame, psi_bins: int = 10,
                 ks_alpha: float = 0.05, psi_threshold: float = 0.2):
        self.reference = reference
        self.psi_bins = psi_bins
        self.ks_alpha = ks_alpha
        self.psi_threshold = psi_threshold
        self.numeric_cols = reference.select_dtypes(include=[np.number]).columns.tolist()

    def _compute_psi(self, ref_col: np.ndarray, curr_col: np.ndarray) -> float:
        """Population Stability Index between two distributions."""
        eps = 1e-4
        # Use reference quantiles as bin edges for consistency
        breakpoints = np.linspace(0, 100, self.psi_bins + 1)
        edges = np.percentile(ref_col, breakpoints)
        edges = np.unique(edges)  # handle constant features
        if len(edges) < 2:
            return 0.0

        # Extend edges to capture current data outside reference range
        edges[0] = min(edges[0], curr_col.min()) - 1
        edges[-1] = max(edges[-1], curr_col.max()) + 1

        ref_counts = np.histogram(ref_col, bins=edges)[0].astype(float)
        curr_counts = np.histogram(curr_col, bins=edges)[0].astype(float)

        # Normalize to proportions
        ref_pct = ref_counts / ref_counts.sum() + eps
        curr_pct = curr_counts / curr_counts.sum() + eps

        psi = np.sum((curr_pct - ref_pct) * np.log(curr_pct / ref_pct))
        return float(psi)

    def _compute_ks(self, ref_col: np.ndarray, curr_col: np.ndarray) -> Dict:
        """KS test between two distributions."""
        stat, p_value = stats.ks_2samp(ref_col, curr_col)
        return {"ks_statistic": float(stat), "p_value": float(p_value),
                "drifted": p_value < self.ks_alpha}

    @timer
    def detect(self, current: pd.DataFrame, features: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Detect drift for each feature.
        Returns DataFrame with columns: feature, psi, ks_statistic, p_value, drifted.
        """
        cols = features or self.numeric_cols
        cols = [c for c in cols if c in current.columns and c in self.reference.columns]

        results = []
        for col in cols:
            ref_vals = self.reference[col].dropna().values
            curr_vals = current[col].dropna().values
            if len(ref_vals) < 2 or len(curr_vals) < 2:
                continue

            psi = self._compute_psi(ref_vals, curr_vals)
            ks = self._compute_ks(ref_vals, curr_vals)
            results.append({
                "feature": col,
                "psi": psi,
                "psi_drifted": psi > self.psi_threshold,
                "ks_statistic": ks["ks_statistic"],
                "p_value": ks["p_value"],
                "ks_drifted": ks["drifted"],
                "drifted": psi > self.psi_threshold or ks["drifted"]
            })

        return pd.DataFrame(results)

    def plot_drift_report(self, report: pd.DataFrame, top_n: int = 20):
        """Plotly bar chart showing PSI per feature, colored by drift status."""
        import plotly.express as px
        df = report.sort_values("psi", ascending=False).head(top_n)
        fig = px.bar(df, x="feature", y="psi", color="drifted",
                     color_discrete_map={True: "#e74c3c", False: "#2ecc71"},
                     title="Data Drift Report (PSI per Feature)")
        fig.add_hline(y=self.psi_threshold, line_dash="dash", line_color="orange",
                      annotation_text=f"Threshold={self.psi_threshold}")
        fig.update_layout(xaxis_tickangle=-45, height=500)
        return fig

    def plot_feature_distribution(self, feature: str, current: pd.DataFrame):
        """Overlay histogram of reference vs current for a single feature."""
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=self.reference[feature].dropna(), name="Reference",
                                   opacity=0.6, nbinsx=40))
        fig.add_trace(go.Histogram(x=current[feature].dropna(), name="Current",
                                   opacity=0.6, nbinsx=40))
        fig.update_layout(barmode="overlay", title=f"Distribution: {feature}",
                          xaxis_title=feature, yaxis_title="Count")
        return fig

    def summary(self, report: pd.DataFrame) -> Dict:
        """Quick summary statistics from a drift report."""
        return {
            "total_features": len(report),
            "drifted_features": int(report["drifted"].sum()),
            "drift_pct": float(report["drifted"].mean() * 100),
            "max_psi": float(report["psi"].max()) if len(report) > 0 else 0.0,
            "worst_feature": report.loc[report["psi"].idxmax(), "feature"] if len(report) > 0 else None,
        }


if __name__ == '__main__':
    # Sample data
    np.random.seed(42)
    size_df = 500

    # Reference (training) data
    reference = pd.DataFrame({
        'feature_a': np.random.randn(size_df),
        'feature_b': np.random.randn(size_df) * 2,
        'feature_c': np.random.exponential(1, size_df),
    })

    # Current (production) data — with drift injected on feature_a
    current = pd.DataFrame({
        'feature_a': np.random.randn(size_df) + 3,  # injected drift
        'feature_b': np.random.randn(size_df) * 2,
        'feature_c': np.random.exponential(1, size_df),
    })

    # Test DriftDetector
    logger.info("🔬 Testing DriftDetector...")
    detector = DriftDetector(reference)
    report = detector.detect(current)
    logger.info(f"✅ Drift report:\n{report}")
    logger.info(f"\n🎯 Summary: {detector.summary(report)}")
