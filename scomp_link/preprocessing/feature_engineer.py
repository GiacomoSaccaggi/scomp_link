# -*- coding: utf-8 -*-
"""
███████╗███████╗ █████╗ ████████╗██╗   ██╗██████╗ ███████╗
██╔════╝██╔════╝██╔══██╗╚══██╔══╝██║   ██║██╔══██╗██╔════╝
█████╗  █████╗  ███████║   ██║   ██║   ██║██████╔╝█████╗  
██╔══╝  ██╔══╝  ██╔══██║   ██║   ██║   ██║██╔══██╗██╔══╝  
██║     ███████╗██║  ██║   ██║   ╚██████╔╝██║  ██║███████╗
╚═╝     ╚══════╝╚═╝  ╚═╝   ╚═╝    ╚═════╝ ╚═╝  ╚═╝╚══════╝

███████╗███╗   ██╗ ██████╗ ██╗███╗   ██╗███████╗███████╗██████╗ 
██╔════╝████╗  ██║██╔════╝ ██║████╗  ██║██╔════╝██╔════╝██╔══██╗
█████╗  ██╔██╗ ██║██║  ██╗ ██║██╔██╗ ██║█████╗  █████╗  ██████╔╝
██╔══╝  ██║╚████║██║  ╚██╗██║██║╚████║██╔══╝  ██╔══╝  ██╔══██╗
███████╗██║ ╚███║╚██████╔╝██║██║ ╚███║███████╗███████╗██║  ██║
╚══════╝╚═╝  ╚══╝ ╚═════╝ ╚═╝╚═╝  ╚══╝╚══════╝╚══════╝╚═╝  ╚═╝
"""
import numpy as np
import pandas as pd
import polars as pl
from typing import Optional, List, Dict, Union
from sklearn.base import BaseEstimator, TransformerMixin

from scomp_link.utils.logger import get_logger
logger = get_logger(__name__)
from scomp_link.utils.decorators import timer, memory_usage



class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Automated feature engineering for tabular data.
    sklearn-compatible (fit/transform). Uses Polars internally for performance.

    Dependencies: numpy, pandas, polars, scikit-learn

    PARAMETERS:
     1. interactions: generate polynomial interaction features (default True)
     2. interaction_degree: max polynomial degree (default 2)
     3. log_transform: apply log1p to skewed numeric features (default True)
     4. skew_threshold: skewness above this triggers log transform (default 1.0)
     5. date_features: extract date components from datetime columns (default True)
     6. target_encode: apply target encoding to high-cardinality categoricals (default True)
     7. target_encode_threshold: cardinality above this triggers encoding (default 10)
     8. auto_bin: bin continuous features into quantile buckets (default False)
     9. n_bins: number of quantile bins (default 5)

    Usage example:
        fe = FeatureEngineer(interactions=True, log_transform=True)
        fe.fit(X_train, y_train)
        X_train_eng = fe.transform(X_train)
        X_test_eng = fe.transform(X_test)
    """

    def __init__(self, interactions: bool = True, interaction_degree: int = 2,
                 log_transform: bool = True, skew_threshold: float = 1.0,
                 date_features: bool = True, target_encode: bool = True,
                 target_encode_threshold: int = 10, auto_bin: bool = False,
                 n_bins: int = 5):
        self.interactions = interactions
        self.interaction_degree = interaction_degree
        self.log_transform = log_transform
        self.skew_threshold = skew_threshold
        self.date_features = date_features
        self.target_encode = target_encode
        self.target_encode_threshold = target_encode_threshold
        self.auto_bin = auto_bin
        self.n_bins = n_bins

        # Fitted state
        self._numeric_cols: List[str] = []
        self._skewed_cols: List[str] = []
        self._date_cols: List[str] = []
        self._high_card_cols: List[str] = []
        self._target_encoding_maps: Dict[str, Dict] = {}
        self._bin_edges: Dict[str, np.ndarray] = {}
        self._interaction_cols: List[tuple] = []

    @timer
    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "FeatureEngineer":
        """Fit the feature engineer on training data. Uses Polars for vectorized analysis."""
        logger.info("🔬 FeatureEngineer: fitting...")

        # Convert to polars for fast analysis
        df_pl = pl.from_pandas(X)

        # Detect numeric and date columns
        self._numeric_cols = [c for c in df_pl.columns
                              if df_pl[c].dtype.is_numeric()]
        self._date_cols = [c for c in df_pl.columns
                           if df_pl[c].dtype in (pl.Date, pl.Datetime)]
        # Also check pandas datetime columns that polars might read as Object/String
        for col in X.select_dtypes(include=['datetime64', 'datetime64[ns]']).columns:
            if col not in self._date_cols:
                self._date_cols.append(col)

        # Detect skewed columns — vectorized with polars
        if self.log_transform and self._numeric_cols:
            skew_df = df_pl.select([
                pl.col(c).skew().alias(c) for c in self._numeric_cols
            ])
            min_df = df_pl.select([
                pl.col(c).min().alias(c) for c in self._numeric_cols
            ])
            for col in self._numeric_cols:
                skew_val = skew_df[col][0]
                min_val = min_df[col][0]
                if skew_val is not None and min_val is not None:
                    if abs(skew_val) > self.skew_threshold and min_val >= 0:
                        self._skewed_cols.append(col)

        # Detect high-cardinality categoricals and compute target encoding maps
        if self.target_encode and y is not None:
            cat_cols = [c for c in df_pl.columns
                        if df_pl[c].dtype in (pl.Utf8, pl.Categorical, pl.String)]
            if cat_cols:
                df_with_y = df_pl.select(cat_cols).with_columns(
                    pl.Series("__target__", y.values if hasattr(y, "values") else y)
                )
                for col in cat_cols:
                    n_unique = df_pl[col].n_unique()
                    if n_unique > self.target_encode_threshold:
                        self._high_card_cols.append(col)
                        mapping_df = df_with_y.group_by(col).agg(
                            pl.col("__target__").mean()
                        )
                        self._target_encoding_maps[col] = dict(
                            zip(mapping_df[col].to_list(), mapping_df["__target__"].to_list())
                        )

        # Compute bin edges
        if self.auto_bin:
            for col in self._numeric_cols:
                try:
                    vals = df_pl[col].drop_nulls().to_numpy()
                    edges = np.percentile(vals, np.linspace(0, 100, self.n_bins + 1))
                    self._bin_edges[col] = np.unique(edges)
                except Exception:
                    pass

        # Interaction column pairs
        if self.interactions and len(self._numeric_cols) >= 2:
            from itertools import combinations
            self._interaction_cols = list(combinations(self._numeric_cols[:10], 2))

        logger.info(f"  ✅ Skewed cols (log): {len(self._skewed_cols)}")
        logger.info(f"  ✅ Date cols: {len(self._date_cols)}")
        logger.info(f"  ✅ Target-encoded cols: {len(self._high_card_cols)}")
        logger.info(f"  ✅ Interaction pairs: {len(self._interaction_cols)}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform data with engineered features. Uses Polars expressions (single pass)."""
        df_pl = pl.from_pandas(X)
        new_exprs: List[pl.Expr] = []
        drop_cols: List[str] = []

        # Log transforms — single batch of expressions
        if self.log_transform:
            for col in self._skewed_cols:
                if col in df_pl.columns:
                    new_exprs.append(pl.col(col).log1p().alias(f"{col}_log"))

        # Date feature extraction
        if self.date_features:
            for col in self._date_cols:
                if col in df_pl.columns:
                    # Ensure column is datetime type
                    if df_pl[col].dtype not in (pl.Date, pl.Datetime):
                        df_pl = df_pl.with_columns(pl.col(col).cast(pl.Datetime))
                    new_exprs.extend([
                        pl.col(col).dt.year().alias(f"{col}_year"),
                        pl.col(col).dt.month().alias(f"{col}_month"),
                        pl.col(col).dt.weekday().alias(f"{col}_day_of_week"),
                        (pl.col(col).dt.weekday() >= 6).cast(pl.Int8).alias(f"{col}_is_weekend"),
                        pl.col(col).dt.quarter().alias(f"{col}_quarter"),
                    ])
                    drop_cols.append(col)

        # Target encoding — vectorized replace
        if self.target_encode:
            for col in self._high_card_cols:
                if col in df_pl.columns:
                    mapping = self._target_encoding_maps[col]
                    global_mean = np.mean(list(mapping.values()))
                    new_exprs.append(
                        pl.col(col).replace_strict(
                            mapping, default=global_mean
                        ).cast(pl.Float64).alias(f"{col}_target_enc")
                    )
                    drop_cols.append(col)

        # Polynomial interactions — vectorized multiplication
        if self.interactions:
            for col_a, col_b in self._interaction_cols:
                if col_a in df_pl.columns and col_b in df_pl.columns:
                    new_exprs.append(
                        (pl.col(col_a) * pl.col(col_b)).alias(f"{col_a}_x_{col_b}")
                    )

        # Apply all expressions in a single pass
        if new_exprs:
            df_pl = df_pl.with_columns(new_exprs)

        # Drop original date/categorical columns
        if drop_cols:
            df_pl = df_pl.drop([c for c in drop_cols if c in df_pl.columns])

        # Auto-binning (post main pass — uses polars cut)
        if self.auto_bin:
            bin_exprs = []
            for col, edges in self._bin_edges.items():
                if col in df_pl.columns and len(edges) > 1:
                    bin_exprs.append(
                        pl.col(col).cut(edges[1:-1].tolist()).cast(pl.Utf8).alias(f"{col}_bin")
                    )
            if bin_exprs:
                df_pl = df_pl.with_columns(bin_exprs)

        return df_pl.to_pandas()

    @memory_usage
    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    def get_feature_names(self, X: pd.DataFrame) -> List[str]:
        """Return list of output feature names after transform."""
        return self.transform(X.head(1)).columns.tolist()


if __name__ == '__main__':
    # Sample data
    np.random.seed(42)
    size_df = 500
    df = pd.DataFrame({
        'income': np.random.exponential(50000, size_df),  # skewed
        'age': np.random.normal(35, 10, size_df),
        'score': np.random.exponential(100, size_df),  # skewed
        'city': np.random.choice(['NYC', 'LA', 'CHI', 'HOU', 'PHX', 'PHI',
                                   'SA', 'SD', 'DAL', 'SJ', 'AUS', 'JAX'], size_df),
        'signup_date': pd.date_range('2020-01-01', periods=size_df, freq='D'),
    })
    y = 0.5 * df['income'] + 100 * df['age'] + np.random.randn(size_df) * 1000

    fe = FeatureEngineer(interactions=True, log_transform=True,
                         date_features=True, target_encode=True, auto_bin=True)
    df_eng = fe.fit_transform(df, y)
    logger.info(f"\n🎯 Original shape: {df.shape} → Engineered shape: {df_eng.shape}")
    logger.info(f"   New columns: {[c for c in df_eng.columns if c not in df.columns]}")
