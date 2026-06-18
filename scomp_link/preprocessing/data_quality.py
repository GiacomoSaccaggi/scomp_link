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
        """Generate a standalone HTML report."""
        if self._report is None:
            self.generate()
        r = self._report
        html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Data Quality Report</title>
<style>
body{{font-family:system-ui;max-width:900px;margin:2rem auto;padding:0 1rem;background:#0f172a;color:#e2e8f0}}
h1{{background:linear-gradient(135deg,#38bdf8,#a78bfa);-webkit-background-clip:text;
-webkit-text-fill-color:transparent;text-align:center}}
h2{{color:#38bdf8;border-bottom:1px solid #334155;padding-bottom:.3rem}}
table{{width:100%;border-collapse:collapse;margin:1rem 0;font-size:.85rem}}
th{{background:#1e293b;color:#94a3b8;text-align:left;padding:.5rem;text-transform:uppercase;font-size:.7rem}}
td{{padding:.4rem .5rem;border-bottom:1px solid #1e293b}}
.metric{{display:inline-block;background:#1e293b;border:1px solid #334155;border-radius:8px;
padding:.8rem 1.2rem;margin:.3rem;text-align:center}}
.metric .val{{font-size:1.4rem;font-weight:700;color:#38bdf8}}
.metric .lbl{{font-size:.7rem;color:#94a3b8;text-transform:uppercase}}
.warn{{color:#fb923c}} .ok{{color:#34d399}}
</style></head><body>
<h1>📊 Data Quality Report</h1>
<div style="text-align:center;margin:1.5rem 0">
<div class="metric"><div class="val">{r['overview']['n_rows']:,}</div><div class="lbl">Rows</div></div>
<div class="metric"><div class="val">{r['overview']['n_cols']}</div><div class="lbl">Columns</div></div>
<div class="metric"><div class="val">{r['overview']['memory_mb']} MB</div><div class="lbl">Memory</div></div>
<div class="metric"><div class="val">{r['duplicates']['n_duplicates']}</div><div class="lbl">Duplicates</div></div>
<div class="metric"><div class="val">{len(r['constants'])}</div><div class="lbl">Constants</div></div>
</div>
<h2>Missing Values</h2>"""
        if len(r['missing']) > 0:
            html += r['missing'].to_html(index=False, classes='')
        else:
            html += '<p class="ok">✅ No missing values</p>'
        html += "<h2>Cardinality</h2>" + r['cardinality'].to_html(index=False, classes='')
        html += "<h2>Type Inference</h2>" + r['types'].to_html(index=False, classes='')
        if len(r['correlations']) > 0:
            html += "<h2>High Correlations (≥0.95)</h2>" + r['correlations'].to_html(index=False, classes='')
        if r['constants']:
            html += f"<h2>Constant Features</h2><p class='warn'>⚠️ {r['constants']}</p>"
        html += "</body></html>"

        with open(path, 'w') as f:
            f.write(html)
        logger.info(f"✅ HTML report saved: {path}")
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
