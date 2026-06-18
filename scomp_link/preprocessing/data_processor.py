# -*- coding: utf-8 -*-
"""

██████╗  █████╗ ████████╗ █████╗ 
██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗
██║  ██║███████║   ██║   ███████║
██║  ██║██╔══██║   ██║   ██╔══██║
██████╔╝██║  ██║   ██║   ██║  ██║
╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝

██████╗ ██████╗ ███████╗██████╗ ██████╗  █████╗  █████╗ ███████╗ ██████╗ ██████╗██╗███╗  ██╗ ██████╗ 
██╔══██╗██╔══██╗██╔════╝██╔══██╗██╔══██╗██╔══██╗██╔══██╗██╔════╝██╔════╝██╔════╝██║████╗ ██║██╔════╝ 
██████╔╝██████╔╝█████╗  ██████╔╝██████╔╝██║  ██║██║  ╚═╝█████╗  ╚█████╗ ╚█████╗ ██║██╔██╗██║██║  ██╗ 
██╔═══╝ ██╔══██╗██╔══╝  ██╔═══╝ ██╔══██╗██║  ██║██║  ██╗██╔══╝   ╚═══██╗ ╚═══██╗██║██║╚████║██║  ╚██╗
██║     ██║  ██║███████╗██║     ██║  ██║╚█████╔╝╚█████╔╝███████╗██████╔╝██████╔╝██║██║ ╚███║╚██████╔╝
╚═╝     ╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝  ╚═╝ ╚════╝  ╚════╝ ╚══════╝╚═════╝ ╚═════╝ ╚═╝╚═╝  ╚══╝ ╚═════╝ 
"""
import polars as pl
import pandas as pd
import numpy as np
from typing import List, Optional, Union

from scomp_link.utils.logger import get_logger
logger = get_logger(__name__)
from scomp_link.utils.decorators import timer



def _to_polars(df: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]) -> pl.DataFrame:
    if isinstance(df, pd.DataFrame):
        return pl.from_pandas(df)
    if isinstance(df, pl.LazyFrame):
        return df.collect()
    return df


class Preprocessor:
    """
    Handles the preprocessing phases (P1-P12) as described in the scomp-link schema.
    Accepts pandas or polars DataFrames as input. Uses polars internally.
    Returns pandas DataFrames for backward compatibility with sklearn.
    """
    def __init__(self, df: Union[pd.DataFrame, pl.DataFrame]):
        self.df = _to_polars(df)
        self.original_df = self.df.clone()

    @timer
    def clean_data(self, remove_outliers: bool = True, outlier_threshold: float = 3.0) -> pd.DataFrame:
        """
        P4: Data Cleaning (removal of formal/logical errors and outliers).
        """
        logger.info("P4: Cleaning data...")
        self.df = self.df.unique()

        if remove_outliers:
            numeric_cols = [c for c, dt in zip(self.df.columns, self.df.dtypes)
                           if dt.is_numeric()]
            for col in numeric_cols:
                mean = self.df[col].mean()
                std = self.df[col].std()
                if std is not None and std > 0:
                    self.df = self.df.filter(
                        ((pl.col(col) - mean) / std).abs() < outlier_threshold
                    )

        logger.info(f"Data cleaned. Current shape: {self.df.shape}")
        return self.df.to_pandas()

    def integrate_data(self, other_df: Union[pd.DataFrame, pl.DataFrame], on: str, how: str = 'left') -> pd.DataFrame:
        """
        P5: Data Integration (combining multiple sources) RECORD LINKAGE.
        """
        logger.info("P5: Integrating data...")
        other = _to_polars(other_df)
        self.df = self.df.join(other, on=on, how=how)
        logger.info(f"Data integrated. Current shape: {self.df.shape}")
        return self.df.to_pandas()

    def transform_data(self) -> pd.DataFrame:
        """
        P7: Data Transformation (reshaping data into forms suitable for data mining).
        """
        logger.info("P7: Transforming data...")
        return self.df.to_pandas()

    def feature_selection(self, target_col: str, n_features: Optional[int] = None) -> List[str]:
        """
        P10: Feature Selection.
        """
        logger.info("P10: Selecting features...")
        if target_col in self.df.columns:
            numeric_cols = [c for c, dt in zip(self.df.columns, self.df.dtypes)
                           if dt.is_numeric() and c != target_col]
            # polars pearson_corr per column
            correlations = {}
            for col in numeric_cols:
                corr = self.df.select(pl.corr(col, target_col)).item()
                if corr is not None:
                    correlations[col] = abs(corr)

            sorted_features = sorted(correlations, key=correlations.get, reverse=True)
            top_features = sorted_features[:n_features] if n_features else sorted_features
            logger.info(f"Top features selected: {top_features}")
            return top_features
        return self.df.columns

    def run_eda(self) -> dict:
        """
        P11: EDA: Knowledge Presentation.
        """
        logger.info("P11: Running Exploratory Data Analysis...")
        null_counts = self.df.null_count().row(0, named=True)
        summary = {
            "shape": self.df.shape,
            "missing_values": null_counts,
            "dtypes": {c: str(dt) for c, dt in zip(self.df.columns, self.df.dtypes)},
            "description": self.df.describe().to_pandas().to_dict()
        }
        return summary

    def prepare_datasets(self, target_col: str, test_size: float = 0.2):
        """
        P12: Dataset Preparation.
        """
        logger.info("P12: Preparing datasets...")
        from sklearn.model_selection import train_test_split

        X = self.df.drop(target_col).to_pandas()
        y = self.df[target_col].to_pandas()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        return X_train, X_test, y_train, y_test
