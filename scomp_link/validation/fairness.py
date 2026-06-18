# -*- coding: utf-8 -*-
"""
███████╗ █████╗ ██╗██████╗ ███╗   ██╗███████╗███████╗███████╗
██╔════╝██╔══██╗██║██╔══██╗████╗  ██║██╔════╝██╔════╝██╔════╝
█████╗  ███████║██║██████╔╝██╔██╗ ██║█████╗  ███████╗███████╗
██╔══╝  ██╔══██║██║██╔══██╗██║╚████║██╔══╝  ╚════██║╚════██║
██║     ██║  ██║██║██║  ██║██║ ╚███║███████╗███████║███████║
╚═╝     ╚═╝  ╚═╝╚═╝╚═╝  ╚═╝╚═╝  ╚══╝╚══════╝╚══════╝╚══════╝
"""
import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Union

from scomp_link.utils.logger import get_logger
logger = get_logger(__name__)
from scomp_link.utils.decorators import timer



class FairnessMetrics:
    """
    Compute fairness and bias metrics for classification models.

    Dependencies: numpy, pandas, plotly

    PARAMETERS:
     1. y_true: ground truth labels (binary 0/1)
     2. y_pred: predicted labels (binary 0/1)
     3. sensitive_feature: array/series of protected attribute values
     4. positive_label: which label is considered "positive" (default 1)

    Usage example:
        fm = FairnessMetrics(y_true, y_pred, sensitive_feature=df['gender'])
        report = fm.compute_all()
        fig = fm.plot_fairness_report(report)
    """

    def __init__(self, y_true, y_pred, sensitive_feature,
                 positive_label: int = 1):
        self.y_true = np.asarray(y_true)
        self.y_pred = np.asarray(y_pred)
        self.sensitive_feature = np.asarray(sensitive_feature)
        self.positive_label = positive_label
        self._groups = np.unique(self.sensitive_feature)

    def _group_mask(self, group) -> np.ndarray:
        return self.sensitive_feature == group

    def demographic_parity(self) -> Dict:
        """
        Demographic Parity: P(Y_hat=1 | A=a) should be equal across groups.
        Returns selection rate per group + ratio (min/max).
        """
        rates = {}
        for g in self._groups:
            mask = self._group_mask(g)
            rates[str(g)] = float((self.y_pred[mask] == self.positive_label).mean())

        ratio = min(rates.values()) / max(rates.values()) if max(rates.values()) > 0 else 0
        return {"selection_rates": rates, "dp_ratio": ratio, "fair": ratio >= 0.8}

    def equalized_odds(self) -> Dict:
        """
        Equalized Odds: P(Y_hat=1 | Y=y, A=a) should be equal across groups for both y=0 and y=1.
        Returns TPR and FPR per group.
        """
        tpr = {}
        fpr = {}
        for g in self._groups:
            mask = self._group_mask(g)
            pos = self.y_true[mask] == self.positive_label
            neg = ~pos
            if pos.sum() > 0:
                tpr[str(g)] = float((self.y_pred[mask][pos] == self.positive_label).mean())
            else:
                tpr[str(g)] = 0.0
            if neg.sum() > 0:
                fpr[str(g)] = float((self.y_pred[mask][neg] == self.positive_label).mean())
            else:
                fpr[str(g)] = 0.0

        tpr_diff = max(tpr.values()) - min(tpr.values())
        fpr_diff = max(fpr.values()) - min(fpr.values())
        return {"tpr": tpr, "fpr": fpr, "tpr_diff": tpr_diff, "fpr_diff": fpr_diff,
                "fair": tpr_diff < 0.1 and fpr_diff < 0.1}

    def disparate_impact(self) -> Dict:
        """
        Disparate Impact: ratio of selection rates between unprivileged and privileged groups.
        The 4/5 rule: ratio should be >= 0.8.
        """
        rates = {}
        for g in self._groups:
            mask = self._group_mask(g)
            rates[str(g)] = float((self.y_pred[mask] == self.positive_label).mean())

        if len(rates) < 2:
            return {"rates": rates, "di_ratio": 1.0, "fair": True}

        # Compare each group against the highest rate
        max_rate = max(rates.values())
        di_ratios = {g: r / max_rate if max_rate > 0 else 1.0 for g, r in rates.items()}
        min_ratio = min(di_ratios.values())
        return {"rates": rates, "di_ratios": di_ratios, "di_ratio": min_ratio,
                "four_fifths_rule": min_ratio >= 0.8}

    def equal_opportunity(self) -> Dict:
        """
        Equal Opportunity: P(Y_hat=1 | Y=1, A=a) should be equal across groups.
        (Only looks at TPR, not FPR.)
        """
        tpr = {}
        for g in self._groups:
            mask = self._group_mask(g)
            pos = self.y_true[mask] == self.positive_label
            if pos.sum() > 0:
                tpr[str(g)] = float((self.y_pred[mask][pos] == self.positive_label).mean())
            else:
                tpr[str(g)] = 0.0

        tpr_diff = max(tpr.values()) - min(tpr.values())
        return {"tpr": tpr, "tpr_diff": tpr_diff, "fair": tpr_diff < 0.1}

    @timer
    def compute_all(self) -> Dict:
        """Compute all fairness metrics at once."""
        logger.info("🔬 Computing fairness metrics...")
        results = {
            "demographic_parity": self.demographic_parity(),
            "equalized_odds": self.equalized_odds(),
            "disparate_impact": self.disparate_impact(),
            "equal_opportunity": self.equal_opportunity(),
        }
        n_fair = sum(1 for v in results.values() if v.get("fair"))
        logger.info(f"  ✅ {n_fair}/{len(results)} metrics pass fairness thresholds")
        return results

    def summary(self, report: Optional[Dict] = None) -> pd.DataFrame:
        """Summary table of fairness metrics."""
        if report is None:
            report = self.compute_all()
        rows = [
            {"metric": "Demographic Parity", "value": report["demographic_parity"]["dp_ratio"],
             "threshold": "≥ 0.8", "fair": report["demographic_parity"]["fair"]},
            {"metric": "Disparate Impact (4/5 rule)", "value": report["disparate_impact"]["di_ratio"],
             "threshold": "≥ 0.8", "fair": report["disparate_impact"]["four_fifths_rule"]},
            {"metric": "Equal Opportunity (TPR diff)", "value": report["equal_opportunity"]["tpr_diff"],
             "threshold": "< 0.1", "fair": report["equal_opportunity"]["fair"]},
            {"metric": "Equalized Odds (TPR diff)", "value": report["equalized_odds"]["tpr_diff"],
             "threshold": "< 0.1", "fair": report["equalized_odds"]["fair"]},
            {"metric": "Equalized Odds (FPR diff)", "value": report["equalized_odds"]["fpr_diff"],
             "threshold": "< 0.1", "fair": report["equalized_odds"]["fair"]},
        ]
        return pd.DataFrame(rows)

    def plot_fairness_report(self, report: Optional[Dict] = None):
        """Plotly grouped bar chart comparing metrics across groups."""
        import plotly.graph_objects as go
        if report is None:
            report = self.compute_all()

        dp_rates = report["demographic_parity"]["selection_rates"]
        tpr = report["equalized_odds"]["tpr"]
        fpr = report["equalized_odds"]["fpr"]

        groups = list(dp_rates.keys())
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Selection Rate', x=groups, y=[dp_rates[g] for g in groups]))
        fig.add_trace(go.Bar(name='TPR', x=groups, y=[tpr[g] for g in groups]))
        fig.add_trace(go.Bar(name='FPR', x=groups, y=[fpr[g] for g in groups]))
        fig.update_layout(barmode='group', title='Fairness Metrics by Group',
                          yaxis_title='Rate', xaxis_title='Group')
        fig.add_hline(y=0.8, line_dash="dash", line_color="orange",
                      annotation_text="4/5 threshold")
        return fig


if __name__ == '__main__':
    # Sample data
    np.random.seed(42)
    n = 1000

    # Simulate a biased classifier
    gender = np.random.choice(['male', 'female'], n)
    y_true = np.random.binomial(1, 0.5, n)

    # Biased predictions: higher acceptance for 'male'
    y_pred = y_true.copy()
    female_mask = gender == 'female'
    flip_idx = np.where(female_mask & (y_true == 1))[0][:50]
    y_pred[flip_idx] = 0  # unfairly reject 50 qualified females

    fm = FairnessMetrics(y_true, y_pred, sensitive_feature=gender)
    report = fm.compute_all()

    logger.info("\n--- Summary ---")
    print(fm.summary(report).to_string(index=False))

    logger.info(f"\n🎯 Demographic Parity ratio: {report['demographic_parity']['dp_ratio']:.3f}")
    logger.info(f"🎯 Disparate Impact ratio: {report['disparate_impact']['di_ratio']:.3f}")
