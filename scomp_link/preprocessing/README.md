# Preprocessing Module

## Overview

The preprocessing module implements the data preparation pipeline (P1-P12) from the analysis workflow, handling data importation, cleaning, integration, transformation, and feature selection.

## Workflow Phases

This module covers phases P1-P12 of the complete analysis workflow:

```
P1: Business/Problem Understanding
P2: Data Understanding
P3: Data Acquisition (internal data, open data, web scraping)
P4: Data Cleaning (removal of formal/logical errors and outliers)
P5: Data Integration (combining different sources - RECORD LINKAGE)
P6: Data Selection (extracting relevant data from database)
P7: Data Transformation (manipulating data for data mining)
P8: Data Mining (applying analytical methods)
P9: Relationship Evaluation (classification of relationships)
P10: Feature Selection
P11: EDA - Knowledge Presentation (visualization and synthesis)
P12: Dataset Preparation (train/test split)
```

## Core Components

### Preprocessor Class

Main class for data preprocessing operations.

```python
from scomp_link.preprocessing import Preprocessor

# Initialize with DataFrame
prep = Preprocessor(df)

# Clean data (P4)
cleaned_df = prep.clean_data(
    remove_outliers=True,
    outlier_threshold=3.0
)

# Integrate external data (P5 - Record Linkage)
external_df = pd.read_csv("external_data.csv")
integrated_df = prep.integrate_data(
    external_df,
    on='id',
    how='left'
)

# Feature selection (P10)
top_features = prep.feature_selection(
    target_col='target',
    n_features=10
)

# EDA - Exploratory Data Analysis (P11)
summary = prep.run_eda()
print(summary['shape'])
print(summary['missing_values'])
print(summary['dtypes'])

# Prepare train/test datasets (P12)
X_train, X_test, y_train, y_test = prep.prepare_datasets(
    target_col='target',
    test_size=0.2
)
```

## Key Features

### 1. Data Cleaning (P4)

Automatic removal of:
- Duplicate records
- Outliers (using z-score or IQR methods)
- Missing values (configurable strategy)
- Formal and logical errors

```python
cleaned_df = prep.clean_data(
    remove_outliers=True,
    outlier_threshold=3.0,
    handle_missing='drop'  # or 'mean', 'median', 'mode'
)
```

### 2. Data Integration (P5)

Record linkage and data fusion from multiple sources:

```python
# Merge datasets
integrated_df = prep.integrate_data(
    external_df,
    on='key_column',
    how='inner'  # or 'left', 'right', 'outer'
)
```

### 3. Feature Selection (P10)

Automatic feature selection using Boruta algorithm:

```python
# Select most important features
selected_features = prep.feature_selection(
    target_col='target',
    n_features=10,
    method='boruta'  # or 'mutual_info', 'chi2'
)
```

### 4. Exploratory Data Analysis (P11)

Comprehensive EDA with statistics and visualizations:

```python
summary = prep.run_eda()

# Access summary statistics
print(summary['shape'])           # Dataset dimensions
print(summary['missing_values'])  # Missing data report
print(summary['dtypes'])          # Data types
print(summary['numeric_summary']) # Descriptive statistics
print(summary['categorical_summary']) # Category counts
```

### 5. Dataset Preparation (P12)

Train/test split with proper preprocessing:

```python
X_train, X_test, y_train, y_test = prep.prepare_datasets(
    target_col='target',
    test_size=0.2,
    random_state=42,
    stratify=True  # For classification
)
```

## Data Transformation (P7)

Automatic handling of different data types:

- **Categorical variables**: One-hot encoding
- **Binary variables**: Binary encoding
- **Numerical variables**: Standardization/normalization
- **Text data**: Tokenization and vectorization
- **Date/time**: Feature extraction (year, month, day, etc.)

## Integration with Pipeline

The Preprocessor is automatically used by `ScompLinkPipeline`:

```python
from scomp_link import ScompLinkPipeline

pipe = ScompLinkPipeline("Analysis Project")

# Automatic preprocessing (P3-P12)
pipe.import_and_clean_data(df)  # P3-P4
pipe.select_variables(target_col='y')  # P6
# Feature selection and EDA happen automatically
```

## Best Practices

1. **Always clean data first**: Run `clean_data()` before other operations
2. **Check for missing values**: Use `run_eda()` to understand data quality
3. **Feature selection**: Use Boruta for automatic feature selection
4. **Outlier handling**: Adjust threshold based on domain knowledge
5. **Record linkage**: Ensure key columns are properly formatted

## Advanced Usage

### Custom Preprocessing Pipeline

```python
# Create custom preprocessing workflow
prep = Preprocessor(df)

# Step 1: Clean
df_clean = prep.clean_data(remove_outliers=True)

# Step 2: Integrate external data
df_integrated = prep.integrate_data(external_df, on='id')

# Step 3: Feature engineering
df_transformed = prep.transform_features(
    date_cols=['date'],
    text_cols=['description']
)

# Step 4: Feature selection
selected_features = prep.feature_selection(
    target_col='target',
    n_features=15
)

# Step 5: Prepare final datasets
X_train, X_test, y_train, y_test = prep.prepare_datasets(
    target_col='target',
    feature_cols=selected_features
)
```

## Dependencies

- pandas: DataFrame operations
- numpy: Numerical computations
- scikit-learn: Preprocessing transformers
- scipy: Statistical functions
- boruta: Feature selection

## See Also

- [Model Selection](../models/README.md)
- [Validation](../validation/README.md)
- [Complete Pipeline](../README.md)
