# Preprocessing

Data cleaning, feature engineering, and quality profiling.

## Modules

| Module | Class | Description |
|--------|-------|-------------|
| `data_processor.py` | `Preprocessor` | Data cleaning (P4-P12), polars backend, outlier removal, EDA |
| `feature_engineer.py` | `FeatureEngineer` | sklearn-compatible: interactions, log transforms, date extraction, target encoding, binning |
| `data_quality.py` | `DataQualityReport` | Profiling: missing values, cardinality, constants, duplicates, correlations, HTML report |

## Usage

```python
from scomp_link import Preprocessor, FeatureEngineer, DataQualityReport

# Clean data
prep = Preprocessor(df)
clean_df = prep.clean_data()

# Engineer features
fe = FeatureEngineer(interactions=True, log_transform=True)
X_eng = fe.fit_transform(X, y)

# Profile data
dqr = DataQualityReport(df)
dqr.generate()
dqr.save_html('report.html')
```
