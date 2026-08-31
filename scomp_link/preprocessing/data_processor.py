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

from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
import polars as pl

from scomp_link.utils.logger import get_logger

logger = get_logger(__name__)
from scomp_link.utils.decorators import timer


def _to_polars(df: Union[pd.DataFrame, pl.DataFrame, pl.LazyFrame]) -> pl.DataFrame:
    if isinstance(df, pd.DataFrame):
        return pl.from_pandas(df)
    if isinstance(df, pl.LazyFrame):
        return df.collect()
    return df


_DROP_FIRST_SUPPORTED: Optional[bool] = None


def _one_hot_encoder():
    """
    Build a OneHotEncoder that drops the first level when the installed
    scikit-learn allows combining ``drop`` with ``handle_unknown="ignore"``.

    Dropping a level matters for linear models: a full one-hot expansion is
    perfectly collinear, which makes coefficients (and therefore SHAP values)
    degenerate even though predictions stay correct. Older scikit-learn versions
    reject the combination, so the capability is probed once and cached.
    """
    global _DROP_FIRST_SUPPORTED
    import pandas as _pd
    from sklearn.preprocessing import OneHotEncoder

    if _DROP_FIRST_SUPPORTED is None:
        try:
            OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False).fit(
                _pd.DataFrame({"_probe": ["a", "b"]})
            )
            _DROP_FIRST_SUPPORTED = True
        except Exception:
            _DROP_FIRST_SUPPORTED = False
            logger.debug("scikit-learn rejects drop='first' with handle_unknown='ignore'; using full one-hot.")

    if _DROP_FIRST_SUPPORTED:
        return OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False)


def build_feature_pipeline(X: "pd.DataFrame | Any"):
    """
    Build a ColumnTransformer that makes any tabular DataFrame model-ready.

    Numeric columns are median-imputed and standardized; categorical columns are
    mode-imputed and one-hot encoded (unknown categories at predict time are
    ignored rather than raising); boolean columns are passed through as floats.

    Returning a transformer rather than a transformed frame lets callers embed it
    in a sklearn Pipeline, so the fitted estimator accepts raw data at predict
    time and artifacts stay self-contained.

    PARAMETERS:
     1. X: feature DataFrame used to infer column groups.

    Usage example:
        from scomp_link.preprocessing.data_processor import build_feature_pipeline
        pre = build_feature_pipeline(X_train)
        model = Pipeline([("preprocessor", pre), ("model", Ridge())])
        model.fit(X_train, y_train)
    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline as SkPipeline
    from sklearn.preprocessing import StandardScaler

    categorical_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    boolean_cols = X.select_dtypes(include=["bool"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()

    transformers = []
    if numeric_cols:
        transformers.append(
            (
                "numeric",
                SkPipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]),
                numeric_cols,
            )
        )
    if categorical_cols:
        transformers.append(
            (
                "categorical",
                SkPipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", _one_hot_encoder()),
                    ]
                ),
                categorical_cols,
            )
        )
    if boolean_cols:
        transformers.append(("boolean", "passthrough", boolean_cols))

    if not transformers:
        raise ValueError("No usable feature columns found (expected numeric, categorical or boolean columns).")

    logger.info(
        f"Feature pipeline: {len(numeric_cols)} numeric, "
        f"{len(categorical_cols)} categorical, {len(boolean_cols)} boolean"
    )
    return ColumnTransformer(transformers=transformers, remainder="drop")


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
            numeric_cols = [c for c, dt in zip(self.df.columns, self.df.dtypes) if dt.is_numeric()]
            for col in numeric_cols:
                mean = self.df[col].mean()
                std = self.df[col].std()
                if std is not None and isinstance(std, (int, float)) and std > 0:
                    self.df = self.df.filter(((pl.col(col) - mean) / std).abs() < outlier_threshold)

        logger.info(f"Data cleaned. Current shape: {self.df.shape}")
        return self.df.to_pandas()

    def integrate_data(self, other_df: Union[pd.DataFrame, pl.DataFrame], on: str, how: str = "left") -> pd.DataFrame:
        """
        P5: Data Integration (combining multiple sources) RECORD LINKAGE.
        """
        from typing import Literal, cast

        logger.info("P5: Integrating data...")
        other = _to_polars(other_df)
        join_how = cast(Literal["inner", "left", "right", "full", "semi", "anti", "cross"], how)
        self.df = self.df.join(other, on=on, how=join_how)
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
            numeric_cols = [c for c, dt in zip(self.df.columns, self.df.dtypes) if dt.is_numeric() and c != target_col]
            # polars pearson_corr per column
            correlations = {}
            for col in numeric_cols:
                corr = self.df.select(pl.corr(col, target_col)).item()
                if corr is not None:
                    correlations[col] = abs(corr)

            sorted_features = sorted(correlations, key=lambda x: correlations[x], reverse=True)
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
            "description": self.df.describe().to_pandas().to_dict(),
        }
        return summary

    def prepare_datasets(self, target_col: str, test_size: float = 0.2, feature_cols: Optional[List[str]] = None):
        """
        P12: Dataset Preparation.

        PARAMETERS:
         1. target_col: name of the target column.
         2. test_size: fraction held out for testing.
         3. feature_cols: explicit feature whitelist. When omitted, every column
            except the target is used. Passing this is what prevents excluded
            columns (leaky identifiers, for instance) from reaching the model.
        """
        logger.info("P12: Preparing datasets...")
        from sklearn.model_selection import train_test_split

        df_pd = self.df.to_pandas()
        if target_col not in df_pd.columns:
            raise ValueError(f"Target column '{target_col}' not found. Available: {list(df_pd.columns)}")

        if feature_cols:
            missing = [c for c in feature_cols if c not in df_pd.columns]
            if missing:
                raise ValueError(f"Feature columns not found in data: {missing}")
            selected = [c for c in feature_cols if c != target_col]
            if not selected:
                raise ValueError("feature_cols must contain at least one column other than the target.")
        else:
            selected = [c for c in df_pd.columns if c != target_col]

        X = df_pd[selected]
        y = df_pd[target_col]
        logger.info(f"Features used ({len(selected)}): {selected}")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        return X_train, X_test, y_train, y_test
