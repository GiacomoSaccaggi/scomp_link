# -*- coding: utf-8 -*-
"""
██████╗  █████╗ ████████╗ █████╗      ██████╗ ██╗   ██╗ █████╗ ██╗     ██╗████████╗██╗   ██╗
██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗    ██╔═══██╗██║   ██║██╔══██╗██║     ██║╚══██╔══╝╚██╗ ██╔╝
██║  ██║███████║   ██║   ███████║    ██║   ██║██║   ██║███████║██║     ██║   ██║    ╚████╔╝ 
██║  ██║██╔══██║   ██║   ██╔══██║    ██║▄▄ ██║██║   ██║██╔══██║██║     ██║   ██║     ╚██╔╝  
██████╔╝██║  ██║   ██║   ██║  ██║    ╚██████╔╝╚██████╔╝██║  ██║███████╗██║   ██║      ██║   
╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝     ╚══▀▀═╝  ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝   ╚═╝      ╚═╝   
"""
import numpy as np
import pandas as pd
from typing import Optional, Dict, List

from scomp_link.utils.logger import get_logger
logger = get_logger(__name__)
from scomp_link.utils.decorators import timer



class DataQualityReport:
    """
    Comprehensive data quality profiling and reporting.

    Dependencies: numpy, pandas, scipy

    PARAMETERS:
     1. df: DataFrame to profile
     2. target_col: optional target column (excluded from some checks)

    Usage example:
        dqr = DataQualityReport(df)
        report = dqr.generate()
        dqr.save_html('data_quality_report.html')
    """

    def __init__(self, df: pd.DataFrame, target_col: Optional[str] = None):
        self.df = df
        self.target_col = target_col
        self._report: Optional[Dict] = None

    @timer
    def generate(self) -> Dict:
        """Run full data quality analysis."""
        logger.info("🔬 DataQualityReport: profiling...")
        self._report = {
            "overview": self._overview(),
            "missing": self._missing_analysis(),
            "types": self._type_inference(),
            "cardinality": self._cardinality_analysis(),
            "constants": self._constant_features(),
            "duplicates": self._duplicate_analysis(),
            "correlations": self._high_correlations(),
        }
        logger.info(f"  ✅ Profiling complete: {self.df.shape[0]} rows × {self.df.shape[1]} cols")
        return self._report

    def _overview(self) -> Dict:
        n_rows, n_cols = self.df.shape
        memory_mb = self.df.memory_usage(deep=True).sum() / 1024 / 1024
        return {"n_rows": n_rows, "n_cols": n_cols, "memory_mb": round(memory_mb, 2),
                "n_numeric": len(self.df.select_dtypes(include=[np.number]).columns),
                "n_categorical": len(self.df.select_dtypes(include=['object', 'category']).columns),
                "n_datetime": len(self.df.select_dtypes(include=['datetime64']).columns)}

    def _missing_analysis(self) -> pd.DataFrame:
        missing = self.df.isnull().sum()
        pct = (missing / len(self.df) * 100).round(2)
        df = pd.DataFrame({"column": missing.index, "missing_count": missing.values,
                           "missing_pct": pct.values})
        df = df[df['missing_count'] > 0].sort_values('missing_pct', ascending=False)
        return df.reset_index(drop=True)

    def _type_inference(self) -> pd.DataFrame:
        rows = []
        for col in self.df.columns:
            dtype = str(self.df[col].dtype)
            inferred = "numeric" if np.issubdtype(self.df[col].dtype, np.number) else dtype
            if dtype == 'object':
                sample = self.df[col].dropna().head(100)
                if sample.str.match(r'^\d{4}-\d{2}-\d{2}').any():
                    inferred = "datetime (string)"
                elif sample.str.match(r'^\d+\.?\d*$').all() and len(sample) > 0:
                    inferred = "numeric (string)"
            rows.append({"column": col, "pandas_dtype": dtype, "inferred_type": inferred})
        return pd.DataFrame(rows)

    def _cardinality_analysis(self) -> pd.DataFrame:
        rows = []
        for col in self.df.columns:
            n_unique = self.df[col].nunique()
            ratio = n_unique / len(self.df)
            flag = ""
            if ratio > 0.95:
                flag = "⚠️ near-unique (possible ID)"
            elif n_unique <= 2:
                flag = "binary"
            elif n_unique <= 10:
                flag = "low-cardinality"
            rows.append({"column": col, "n_unique": n_unique, "uniqueness_ratio": round(ratio, 4), "flag": flag})
        return pd.DataFrame(rows).sort_values("n_unique", ascending=False).reset_index(drop=True)

    def _constant_features(self) -> List[str]:
        constants = []
        for col in self.df.columns:
            if self.df[col].nunique(dropna=True) <= 1:
                constants.append(col)
        return constants

    def _duplicate_analysis(self) -> Dict:
        n_dupes = self.df.duplicated().sum()
        return {"n_duplicates": int(n_dupes), "duplicate_pct": round(n_dupes / len(self.df) * 100, 2)}

    def _high_correlations(self, threshold: float = 0.95) -> pd.DataFrame:
        numeric = self.df.select_dtypes(include=[np.number])
        if numeric.shape[1] < 2:
            return pd.DataFrame(columns=["col_a", "col_b", "correlation"])
        corr = numeric.corr().abs()
        pairs = []
        for i in range(len(corr.columns)):
            for j in range(i + 1, len(corr.columns)):
                if corr.iloc[i, j] >= threshold:
                    pairs.append({"col_a": corr.columns[i], "col_b": corr.columns[j],
                                  "correlation": round(corr.iloc[i, j], 4)})
        df = pd.DataFrame(pairs)
        if len(df) == 0:
            return df
        return df.sort_values("correlation", ascending=False).reset_index(drop=True)

    def save_html(self, path: str = "data_quality_report.html"):
        """Generate a standalone HTML report using ScompLinkHTMLReport."""
        from scomp_link.utils.report_html import ScompLinkHTMLReport

        if self._report is None:
            self.generate()
        r = self._report

        report = ScompLinkHTMLReport('Data Quality Report')
        report.open_section('Overview')
        overview_df = pd.DataFrame([r['overview']])
        report.add_dataframe(overview_df, 'overview_metrics')
        report.add_text(f"Duplicates: {r['duplicates']['n_duplicates']} ({r['duplicates']['duplicate_pct']}%)")
        if r['constants']:
            report.add_text(f"Constant features: {r['constants']}")
        report.close_section()

        report.open_section('Missing Values')
        if len(r['missing']) > 0:
            report.add_dataframe(r['missing'], 'missing_values')
        else:
            report.add_text('No missing values detected.')
        report.close_section()

        report.open_section('Cardinality')
        report.add_dataframe(r['cardinality'], 'cardinality')
        report.close_section()

        report.open_section('Type Inference')
        report.add_dataframe(r['types'], 'type_inference')
        report.close_section()

        if len(r['correlations']) > 0:
            report.open_section('High Correlations (>=0.95)')
            report.add_dataframe(r['correlations'], 'high_correlations')
            report.close_section()

        report.save_html(path)
        logger.info(f"\u2705 HTML report saved: {path}")
        return path


if __name__ == '__main__':
    # Sample data
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        'id': range(n),
        'age': np.random.normal(35, 10, n),
        'income': np.random.lognormal(10, 0.5, n),
        'score': np.random.normal(700, 50, n),
        'category': np.random.choice(['A', 'B', 'C'], n),
        'constant_col': 'same_value',
        'mostly_null': np.where(np.random.rand(n) > 0.1, np.nan, 1.0),
    })
    # Add some duplicates
    df = pd.concat([df, df.head(20)], ignore_index=True)

    dqr = DataQualityReport(df)
    report = dqr.generate()
    logger.info(f"\n🎯 Overview: {report['overview']}")
    logger.info(f"🎯 Duplicates: {report['duplicates']}")
    logger.info(f"🎯 Constants: {report['constants']}")
