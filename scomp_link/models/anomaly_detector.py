# -*- coding: utf-8 -*-
"""

 █████╗ ███╗   ██╗ ██████╗ ███╗   ███╗ █████╗ ██╗  ██╗   ██╗
██╔══██╗████╗  ██║██╔═══██╗████╗ ████║██╔══██╗██║  ╚██╗ ██╔╝
███████║██╔██╗ ██║██║   ██║██╔████╔██║███████║██║   ╚████╔╝ 
██╔══██║██║╚██╗██║██║   ██║██║╚██╔╝██║██╔══██║██║    ╚██╔╝  
██║  ██║██║ ╚████║╚██████╔╝██║ ╚═╝ ██║██║  ██║███████╗██║   
╚═╝  ╚═╝╚═╝  ╚═══╝ ╚═════╝ ╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝╚═╝   

██████╗ ███████╗████████╗███████╗ ██████╗████████╗ ██████╗ ██████╗ 
██╔══██╗██╔════╝╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝██╔═══██╗██╔══██╗
██║  ██║█████╗     ██║   █████╗  ██║        ██║   ██║   ██║██████╔╝
██║  ██║██╔══╝     ██║   ██╔══╝  ██║        ██║   ██║   ██║██╔══██╗
██████╔╝███████╗   ██║   ███████╗╚██████╗   ██║   ╚██████╔╝██║  ██║
╚═════╝ ╚══════╝   ╚═╝   ╚══════╝ ╚═════╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝

"""

import numpy as np
import pandas as pd
import warnings
from typing import Optional, Dict, List
from sklearn.base import BaseEstimator
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler

from scomp_link.utils.logger import get_logger
logger = get_logger(__name__)
from scomp_link.utils.decorators import timer


warnings.filterwarnings("ignore", category=UserWarning)


class AnomalyDetector(BaseEstimator):
    """
    Multi-method anomaly detection for tabular data.

    Implements 4 complementary approaches:
      1. Isolation Forest — tree-based, non-parametric
      2. Local Outlier Factor — density-based, local neighborhoods
      3. TabNet Autoencoder — neural attention on tabular features (pytorch-tabnet)
      4. Transformer Autoencoder — self-attention across features as tokens

    Uses consensus voting (majority rule) to produce robust anomaly labels.

    Parameters
    ----------
    contamination : float, default=0.05
        Expected proportion of anomalies in the dataset.
    methods : list of str, optional
        Which methods to run. Default: all four.
        Options: 'iforest', 'lof', 'tabnet', 'transformer'
    lof_neighbors : int, default=20
        Number of neighbors for LOF.
    tabnet_epochs : int, default=50
        Max training epochs for TabNet pretrainer.
    transformer_epochs : int, default=80
        Training epochs for Transformer autoencoder.
    transformer_d_model : int, default=32
        Embedding dimension for Transformer.
    transformer_nhead : int, default=4
        Number of attention heads.
    transformer_num_layers : int, default=2
        Number of Transformer encoder layers.
    consensus_threshold : int, default=2
        Minimum number of methods that must agree to flag an anomaly.
    random_state : int, default=42
        Random seed for reproducibility.
    verbose : bool, default=True
        Print progress messages.

    Example
    -------
    >>> from scomp_link.models.anomaly_detector import AnomalyDetector
    >>> import pandas as pd
    >>>
    >>> df = pd.read_csv("data.csv")
    >>> detector = AnomalyDetector(contamination=0.05)
    >>> results = detector.fit_predict(df, features=['col1', 'col2', 'col3'])
    >>> print(results['comparison'])
    >>> anomalies = results['data'][results['data']['is_anomaly']]
    """

    AVAILABLE_METHODS = ['iforest', 'lof', 'tabnet', 'transformer']

    def __init__(
        self,
        contamination: float = 0.05,
        methods: Optional[List[str]] = None,
        lof_neighbors: int = 20,
        tabnet_epochs: int = 50,
        transformer_epochs: int = 80,
        transformer_d_model: int = 32,
        transformer_nhead: int = 4,
        transformer_num_layers: int = 2,
        consensus_threshold: int = 2,
        random_state: int = 42,
        verbose: bool = True,
    ):
        self.contamination = contamination
        self.methods = methods or self.AVAILABLE_METHODS
        self.lof_neighbors = lof_neighbors
        self.tabnet_epochs = tabnet_epochs
        self.transformer_epochs = transformer_epochs
        self.transformer_d_model = transformer_d_model
        self.transformer_nhead = transformer_nhead
        self.transformer_num_layers = transformer_num_layers
        self.consensus_threshold = consensus_threshold
        self.random_state = random_state
        self.verbose = verbose

        self.scaler_ = None
        self.results_ = None

    def _log(self, msg: str):
        if self.verbose:
            logger.info(msg)

    # ------------------------------------------------------------------
    # 1. Isolation Forest
    # ------------------------------------------------------------------

    def _run_iforest(self, X: np.ndarray) -> np.ndarray:
        """
        Isolation Forest: isolates anomalies via random recursive partitioning.
        Non-parametric — no normality assumption on heavy-tailed data.
        """
        self._log("  [1/4] Isolation Forest...")
        model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        )
        return model.fit_predict(X) == -1

    # ------------------------------------------------------------------
    # 2. Local Outlier Factor
    # ------------------------------------------------------------------

    def _run_lof(self, X: np.ndarray) -> np.ndarray:
        """
        LOF: measures local density deviation.
        Catches anomalies that live in sparse neighborhoods relative to their peers.
        """
        self._log("  [2/4] Local Outlier Factor...")
        model = LocalOutlierFactor(
            n_neighbors=self.lof_neighbors,
            contamination=self.contamination,
            n_jobs=-1,
        )
        return model.fit_predict(X) == -1

    # ------------------------------------------------------------------
    # 3. TabNet Autoencoder
    # ------------------------------------------------------------------

    def _run_tabnet(self, X: np.ndarray) -> np.ndarray:
        """
        TabNet unsupervised pretrainer as autoencoder.
        Uses attention mechanism on tabular features — learns which features
        matter for reconstruction. High reconstruction error → anomaly.
        Requires: pytorch-tabnet
        """
        self._log("  [3/4] TabNet Autoencoder...")
        try:
            from pytorch_tabnet.pretraining import TabNetPretrainer
        except ImportError:
            raise ImportError(
                "pytorch-tabnet required for TabNet anomaly detection. "
                "Install with: pip install pytorch-tabnet"
            )

        X_float = X.astype(np.float32)
        pretrainer = TabNetPretrainer(
            n_d=8, n_a=8, n_steps=3,
            optimizer_params=dict(lr=2e-2),
            mask_type="entmax",
            verbose=0,
            seed=self.random_state,
        )
        pretrainer.fit(
            X_train=X_float,
            eval_set=[X_float],
            max_epochs=self.tabnet_epochs,
            patience=10,
            batch_size=256,
            pretraining_ratio=0.5,
        )

        reconstructed, _ = pretrainer.predict(X_float)
        recon_error = np.mean((X_float - reconstructed) ** 2, axis=1)
        threshold = np.percentile(recon_error, (1 - self.contamination) * 100)
        return recon_error > threshold

    # ------------------------------------------------------------------
    # 4. Transformer Autoencoder
    # ------------------------------------------------------------------

    def _run_transformer(self, X: np.ndarray) -> np.ndarray:
        """
        Transformer autoencoder for tabular data.
        Each feature is treated as a token in a sequence — self-attention
        captures inter-feature relationships. High reconstruction error → anomaly.
        Requires: torch
        """
        self._log("  [4/4] Transformer Autoencoder...")
        try:
            import torch
            import torch.nn as nn
            from torch.utils.data import DataLoader, TensorDataset
        except ImportError:
            raise ImportError(
                "PyTorch required for Transformer anomaly detection. "
                "Install with: pip install torch"
            )

        X_float = X.astype(np.float32)
        n_features = X_float.shape[1]
        d_model = self.transformer_d_model
        nhead = self.transformer_nhead
        num_layers = self.transformer_num_layers

        class _TabularTransformerAE(nn.Module):
            def __init__(self_model):
                super().__init__()
                self_model.input_proj = nn.Linear(1, d_model)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=nhead,
                    dim_feedforward=d_model * 4,
                    dropout=0.1, batch_first=True,
                )
                self_model.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self_model.decoder = nn.Linear(d_model, 1)

            def forward(self_model, x):
                x = x.unsqueeze(-1)
                x = self_model.input_proj(x)
                x = self_model.encoder(x)
                x = self_model.decoder(x).squeeze(-1)
                return x

        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        torch.manual_seed(self.random_state)

        model = _TabularTransformerAE().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()

        dataset = TensorDataset(torch.tensor(X_float))
        loader = DataLoader(dataset, batch_size=256, shuffle=True)

        model.train()
        for _ in range(self.transformer_epochs):
            for (batch,) in loader:
                batch = batch.to(device)
                recon = model(batch)
                loss = criterion(recon, batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_float).to(device)
            recon = model(X_tensor).cpu().numpy()

        recon_error = np.mean((X_float - recon) ** 2, axis=1)
        threshold = np.percentile(recon_error, (1 - self.contamination) * 100)
        return recon_error > threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @timer
    def fit_predict(
        self,
        df: pd.DataFrame,
        features: List[str],
    ) -> Dict:
        """
        Run all configured anomaly detection methods and produce consensus.

        Parameters
        ----------
        df : pd.DataFrame
            Input dataframe.
        features : list of str
            Numeric columns to use for detection.

        Returns
        -------
        dict with keys:
            - 'data': DataFrame with anomaly columns added
            - 'comparison': DataFrame summarizing each method
            - 'consensus_threshold': int
        """
        clean = df.dropna(subset=features).copy()
        self.scaler_ = StandardScaler()
        X = self.scaler_.fit_transform(clean[features])

        method_map = {
            'iforest': self._run_iforest,
            'lof': self._run_lof,
            'tabnet': self._run_tabnet,
            'transformer': self._run_transformer,
        }

        anom_cols = []
        for method in self.methods:
            if method not in method_map:
                raise ValueError(f"Unknown method '{method}'. Available: {self.AVAILABLE_METHODS}")
            col_name = f"anom_{method}"
            clean[col_name] = method_map[method](X)
            anom_cols.append(col_name)

        # Consensus
        clean["consensus_score"] = clean[anom_cols].sum(axis=1)
        clean["is_anomaly"] = clean["consensus_score"] >= self.consensus_threshold

        # Comparison table
        comparison = pd.DataFrame({
            "method": [c.replace("anom_", "") for c in anom_cols],
            "n_anomalies": [int(clean[c].sum()) for c in anom_cols],
            "pct": [clean[c].mean() * 100 for c in anom_cols],
        })
        comparison.loc[len(comparison)] = [
            f"consensus (≥{self.consensus_threshold})",
            int(clean["is_anomaly"].sum()),
            clean["is_anomaly"].mean() * 100,
        ]

        self._log(f"\n  Consensus anomalies: {int(clean['is_anomaly'].sum())} / {len(clean)}")

        self.results_ = {
            "data": clean,
            "comparison": comparison,
            "consensus_threshold": self.consensus_threshold,
        }
        return self.results_

    def report(self, group_by: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Generate a summary report of detected anomalies.

        Parameters
        ----------
        group_by : list of str, optional
            Columns to group anomalies by. If None, returns raw anomalies.

        Returns
        -------
        pd.DataFrame
        """
        if self.results_ is None:
            raise ValueError("No results. Call fit_predict() first.")

        data = self.results_["data"]
        anomalies = data[data["is_anomaly"]]

        if anomalies.empty:
            return pd.DataFrame()

        if group_by is None:
            return anomalies

        agg_dict = {"is_anomaly": "sum", "consensus_score": "mean"}
        # Add numeric columns that exist
        for col in data.select_dtypes(include=[np.number]).columns:
            if col not in ["is_anomaly", "consensus_score"] and not col.startswith("anom_"):
                agg_dict[col] = "mean"

        return (
            anomalies.groupby(group_by)
            .agg(**{
                "count": ("is_anomaly", "sum"),
                "avg_consensus": ("consensus_score", "mean"),
            })
            .sort_values("count", ascending=False)
        )
