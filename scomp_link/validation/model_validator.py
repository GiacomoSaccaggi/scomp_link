# -*- coding: utf-8 -*-
"""

██╗   ██╗ ██████╗ ██╗     ██╗██████╗  ██████╗ ███████╗ ██████╗ ██████╗ 
██║   ██║██╔════╝██║     ██║██╔═══██╗██╔═══██╗██╔════╝██╔════╝██╔══██╗
██║   ██║█████╗  ██║     ██║██║   ██║███████║███████╗██║   ██╗██████╔╝
╚██╗ ██╔╝██╔══╝  ██║     ██║██║   ██║██╔══██║╚════██║██║   ██║██╔══██╗
 ╚████╔╝ ███████╗███████╗██║╚██████╔╝██████╔╝██║███████║╚██████╔╝██║  ██║
  ╚═══╝  ╚══════╝╚══════╝╚═╝ ╚═════╝ ╚═════╝ ╚═╝╚══════╝ ╚═════╝ ╚═╝  ╚═╝

"""
from sklearn.model_selection import LeaveOneOut, KFold, cross_val_score
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from typing import Any, Dict, Optional
from ..utils.report_html import ScompLinkHTMLReport

class Validator:
    """
    Handles modeling and validation phases (M4, C1-C4, V1-V3).
    """
    def __init__(self, model: Any):
        self.model = model

    def k_fold_cv(self, X, y, k: int = 5):
        """
        C2: K-Fold Cross Validation.
        """
        print(f"C2: Running {k}-Fold Cross Validation...")
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        scores = cross_val_score(self.model, X, y, cv=kf)
        print(f"Mean CV Score: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
        return scores

    def loocv(self, X, y):
        """
        C1: Leave-one-out Cross Validation LOOCV.
        """
        print("C1: Running LOOCV (this might be slow)...")
        loo = LeaveOneOut()
        scores = cross_val_score(self.model, X, y, cv=loo)
        print(f"Mean LOOCV Score: {np.mean(scores):.4f}")
        return scores

    def evaluate(self, y_true, y_pred, task_type: str = "regression", y_proba: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        V3: Metriche per valutare i modelli.
        """
        print(f"V3: Evaluating metrics for {task_type}...")
        metrics = {}
        if task_type == "regression":
            from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
            metrics["mse"] = float(mean_squared_error(y_true, y_pred))
            metrics["rmse"] = float(np.sqrt(metrics["mse"]))
            metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
            metrics["r2"] = float(r2_score(y_true, y_pred))
        elif task_type == "classification":
            from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
            metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
            metrics["f1"] = float(f1_score(y_true, y_pred, average='weighted'))
            metrics["precision"] = float(precision_score(y_true, y_pred, average='weighted'))
            metrics["recall"] = float(recall_score(y_true, y_pred, average='weighted'))
        
        for name, value in metrics.items():
            print(f" - {name}: {value:.4f}")
        return metrics

    def generate_validation_report(self, y_true, y_pred, task_type: str = "regression", 
                                  y_proba: Optional[np.ndarray] = None, 
                                  report_name: str = "Validation_Report.html"):
        """
        Generates an HTML report with validation graphs.
        """
        print(f"Generating {task_type} validation report...")
        report = ScompLinkHTMLReport(title=f"Model Validation Report - {task_type.capitalize()}")
        
        metrics = self.evaluate(y_true, y_pred, task_type=task_type)
        
        report.open_section("Metrics Summary")
        metrics_df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
        report.add_dataframe(metrics_df, "Validation Metrics")
        report.close_section()

        if task_type == "regression":
            report.open_section("Regression Analysis")
            
            # 1. Observed vs Predicted
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(x=y_true, y=y_pred, mode='markers', name='Predictions',
                                     marker=dict(color='#6E37FA', opacity=0.5)))
            fig1.add_trace(go.Scatter(x=[y_true.min(), y_true.max()], y=[y_true.min(), y_true.max()],
                                     mode='lines', name='Ideal', line=dict(color='red', dash='dash')))
            fig1.update_layout(title="Observed vs Predicted", xaxis_title="Observed", yaxis_title="Predicted")
            report.add_graph_to_report(fig1, "Observed vs Predicted")
            
            # 2. Residuals Distribution
            residuals = y_pred - y_true
            fig2 = go.Figure()
            fig2.add_trace(go.Histogram(x=residuals, name='Residuals', marker_color='#32BBB9'))
            fig2.update_layout(title="Residuals Distribution", xaxis_title="Residual", yaxis_title="Count")
            report.add_graph_to_report(fig2, "Residuals Distribution")
            
            report.close_section()
            
        elif task_type == "classification":
            report.open_section("Classification Analysis")
            
            # 1. Confusion Matrix
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_true, y_pred)
            labels = sorted(list(set(y_true) | set(y_pred)))
            
            fig1 = go.Figure(data=go.Heatmap(
                z=cm, x=labels, y=labels,
                colorscale='Viridis',
                text=cm, texttemplate="%{text}",
                hoverinfo="z"
            ))
            fig1.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="True")
            report.add_graph_to_report(fig1, "Confusion Matrix")
            
            # 2. Probability distribution if available
            if y_proba is not None:
                fig2 = go.Figure()
                for i, label in enumerate(labels):
                    if i < y_proba.shape[1]:
                        fig2.add_trace(go.Histogram(x=y_proba[:, i], name=f"Prob {label}", opacity=0.6))
                fig2.update_layout(title="Probability Distributions", barmode='overlay', 
                                  xaxis_title="Confidence", yaxis_title="Count")
                report.add_graph_to_report(fig2, "Confidence Distribution")
                
            report.close_section()

        report.save_html(report_name)
        print(f"Report saved to {report_name}")
