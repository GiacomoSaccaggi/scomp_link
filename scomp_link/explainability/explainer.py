# -*- coding: utf-8 -*-
"""
███████╗██╗  ██╗██████╗ ██╗      █████╗ ██╗███╗   ██╗
██╔════╝╚██╗██╔╝██╔══██╗██║     ██╔══██╗██║████╗  ██║
█████╗   ╚███╔╝ ██████╔╝██║     ███████║██║██╔██╗ ██║
██╔══╝   ██╔██╗ ██╔═══╝ ██║     ██╔══██║██║██║╚████║
███████╗██╔╝ ██╗██║     ███████╗██║  ██║██║██║ ╚███║
╚══════╝╚═╝  ╚═╝╚═╝     ╚══════╝╚═╝  ╚═╝╚═╝╚═╝  ╚══╝
"""
import numpy as np
import pandas as pd
from typing import Optional, Union, List

from scomp_link.utils.logger import get_logger
logger = get_logger(__name__)
from scomp_link.utils.decorators import timer



class ShapExplainer:
    """
    Model-agnostic SHAP explanations for any sklearn-compatible estimator.
    
    Dependencies: shap, plotly
    
    PARAMETERS:
     1. model: fitted sklearn-compatible estimator
     2. X_background: background dataset for SHAP (subset of training data)
    
    Usage example:
        explainer = ShapExplainer(model, X_train[:100])
        shap_values = explainer.explain(X_test)
        fig = explainer.plot_importance()
    """

    def __init__(self, model, X_background: pd.DataFrame):
        import shap
        self.model = model
        self.X_background = X_background
        self.feature_names = list(X_background.columns) if hasattr(X_background, 'columns') else None
        self.explainer = shap.Explainer(model, X_background)
        self.shap_values_ = None

    @timer
    def explain(self, X: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values for X. Returns array of shape (n_samples, n_features)."""
        import shap
        self.shap_values_ = self.explainer(X)
        return self.shap_values_.values

    def feature_importance(self) -> pd.DataFrame:
        """Mean absolute SHAP value per feature, sorted descending."""
        if self.shap_values_ is None:
            raise ValueError("Call explain() first.")
        importance = np.abs(self.shap_values_.values).mean(axis=0)
        if importance.ndim > 1:
            importance = importance.mean(axis=-1)
        names = self.feature_names or [f"f{i}" for i in range(len(importance))]
        df = pd.DataFrame({"feature": names, "mean_abs_shap": importance})
        return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    def plot_importance(self, top_n: int = 20):
        """Plotly horizontal bar chart of feature importance."""
        import plotly.express as px
        df = self.feature_importance().head(top_n).iloc[::-1]
        fig = px.bar(df, x="mean_abs_shap", y="feature", orientation="h",
                     title="SHAP Feature Importance", labels={"mean_abs_shap": "Mean |SHAP|"})
        fig.update_layout(height=max(400, top_n * 25))
        return fig

    def plot_beeswarm(self):
        """Matplotlib beeswarm plot (classic SHAP visualization)."""
        import shap
        if self.shap_values_ is None:
            raise ValueError("Call explain() first.")
        shap.plots.beeswarm(self.shap_values_, show=False)

    def plot_waterfall(self, idx: int = 0):
        """Waterfall plot for a single prediction."""
        import shap
        if self.shap_values_ is None:
            raise ValueError("Call explain() first.")
        shap.plots.waterfall(self.shap_values_[idx], show=False)


class LimeExplainer:
    """
    LIME explanations for tabular data.
    
    Dependencies: lime
    
    PARAMETERS:
     1. model: fitted sklearn-compatible estimator
     2. X_train: training data (used to fit LIME's internal kernel)
     3. task: 'regression' or 'classification'
     4. feature_names: optional list of feature names
    
    Usage example:
        explainer = LimeExplainer(model, X_train, task='regression')
        exp = explainer.explain_instance(X_test.iloc[0])
        fig = explainer.plot_explanation(exp)
    """

    def __init__(self, model, X_train: pd.DataFrame, task: str = "regression",
                 feature_names: Optional[List[str]] = None):
        import lime.lime_tabular
        self.model = model
        self.task = task
        self.feature_names = feature_names or (list(X_train.columns) if hasattr(X_train, 'columns') else None)
        mode = "regression" if task == "regression" else "classification"
        X_arr = X_train.values if hasattr(X_train, 'values') else np.array(X_train)
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            X_arr, feature_names=self.feature_names, mode=mode, verbose=False
        )

    def explain_instance(self, instance, num_features: int = 10):
        """Explain a single instance. Returns lime Explanation object."""
        x = instance.values if hasattr(instance, 'values') else np.array(instance)
        predict_fn = self.model.predict_proba if self.task != "regression" else self.model.predict
        return self.explainer.explain_instance(x, predict_fn, num_features=num_features)

    def plot_explanation(self, explanation, top_n: int = 10):
        """Plotly bar chart from a LIME explanation."""
        import plotly.graph_objects as go
        features_weights = explanation.as_list()[:top_n]
        features = [fw[0] for fw in features_weights][::-1]
        weights = [fw[1] for fw in features_weights][::-1]
        colors = ["#e74c3c" if w < 0 else "#2ecc71" for w in weights]
        fig = go.Figure(go.Bar(x=weights, y=features, orientation="h", marker_color=colors))
        fig.update_layout(title="LIME Explanation", xaxis_title="Weight", height=max(350, top_n * 30))
        return fig

    def feature_importance(self, X: pd.DataFrame, n_samples: int = 50, num_features: int = 10) -> pd.DataFrame:
        """Aggregate LIME importance over multiple samples."""
        indices = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
        importance_acc = {}
        for i in indices:
            row = X.iloc[i] if hasattr(X, 'iloc') else X[i]
            exp = self.explain_instance(row, num_features=num_features)
            for feat, weight in exp.as_list():
                importance_acc.setdefault(feat, []).append(abs(weight))
        df = pd.DataFrame([
            {"feature": k, "mean_abs_weight": np.mean(v)}
            for k, v in importance_acc.items()
        ])
        return df.sort_values("mean_abs_weight", ascending=False).reset_index(drop=True)


if __name__ == '__main__':
    # Sample data
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split

    size_df = 300
    X = pd.DataFrame({
        'x1': np.random.randn(size_df),
        'x2': np.random.randn(size_df),
        'x3': np.random.randn(size_df),
    })
    y = 2 * X['x1'] + 0.5 * X['x2'] + np.random.randn(size_df) * 0.1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    # Test ShapExplainer
    logger.info("🔬 Testing ShapExplainer...")
    shap_exp = ShapExplainer(model, X_train[:50])
    shap_exp.explain(X_test[:20])
    importance = shap_exp.feature_importance()
    logger.info(f"✅ SHAP importance:\n{importance}")

    # Test LimeExplainer
    logger.info("\n🔬 Testing LimeExplainer...")
    lime_exp = LimeExplainer(model, X_train, task='regression')
    exp = lime_exp.explain_instance(X_test.iloc[0])
    logger.info(f"✅ LIME explanation: {exp.as_list()[:3]}")
