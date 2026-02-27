# scomp-link: The Astromech Arm for Your Python Projects

## May the code be with you

[![Tests](https://img.shields.io/badge/tests-41%2F41%20passing-brightgreen)]()
[![Coverage](https://img.shields.io/badge/coverage-~100%25-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.7%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()

## Overview

**scomp-link** is a general-purpose machine learning toolkit that automates the complete ML workflow from problem identification to model validation. It implements a comprehensive decision-tree-based analysis workflow covering all phases from data preprocessing (P1-P12) to model selection, training, validation, and ensemble learning.

### Complete Analysis Workflow

The package implements the full data science workflow:

```
PROBLEM IDENTIFICATION → OBJECTIVES FORMULATION → ANALYSIS DEVELOPMENT
    ↓
PREPROCESSING (P1-P12):
  P1: Business/Problem Understanding
  P2: Data Understanding
  P3: Data Acquisition
  P4: Data Cleaning
  P5: Data Integration (Record Linkage)
  P6: Data Selection
  P7: Data Transformation
  P8: Data Mining
  P9: Relationship Evaluation
  P10: Feature Selection
  P11: EDA (Exploratory Data Analysis)
  P12: Dataset Preparation
    ↓
MODEL SELECTION (Decision Tree):
  - Numerical Prediction (< 1k, 1k-100k, > 100k records)
  - Categorical Classification (Images, Categorical, Mixed)
  - Clustering (Known/Unknown categories)
  - Time Series (UCM, VAR/VARMA)
  - Multi-target Prediction
    ↓
MODELING (M1-M4):
  M1: Missing Values Handling
  M2: Outlier Management
  M3: Algorithm Parameters
  M4: Validation Parameters (LOOCV, K-Fold, Bootstrap)
    ↓
VALIDATION:
  V1: Interpretation vs Flexibility
  V2: Underfitting vs Overfitting
  V3: Evaluation Metrics
    ↓
FAIL → Return to Model Selection
SUCCESS → Ensemble Learning → Reinforcement Learning
```

### Key Features

- 🚀 **End-to-End Automation**: Complete workflow from problem to solution
- 🎯 **Multi-Modal Support**: Tabular, text, and image data
- 🧠 **Intelligent Model Selection**: Decision-tree-based algorithm selection
- 🔄 **Advanced Validation**: LOOCV, Bootstrap, K-Fold CV
- 🎭 **Ensemble Learning**: Voting and stacking strategies
- 🌐 **Domain Agnostic**: No hard-coded assumptions
- 🔌 **Pluggable Architecture**: Optional dependencies loaded on-demand
- 📊 **Automated Reporting**: Interactive HTML reports with Plotly

---

## Installation

### Basic Installation

```bash
# Clone the repository
git clone <repository-url>
cd scomp_link

# Install core dependencies
pip install -r requirements.txt

# Or install as package
pip install .
```

### Optional Features

```bash
# Install with NLP support (torch, transformers, spacy)
pip install .[nlp]

# Install with computer vision support (tensorflow, pillow)
pip install .[img]

# Install with utility packages (tqdm, PyJWT)
pip install .[utils]

# Install ALL optional dependencies (includes contrastive learning)
pip install .[all]
```

**Note**: For contrastive text classification, install NLP dependencies:
```bash
pip install torch transformers
pip install faiss-cpu  # Optional, for fast inference
```

---

## Quick Start

### 1. Basic Regression Pipeline

```python
from scomp_link import ScompLinkPipeline
import pandas as pd
import numpy as np

# Create synthetic data
N = 1000
df = pd.DataFrame({
    'x1': np.random.randn(N),
    'x2': np.random.randn(N),
    'y':  2*np.random.randn(N) + 0.5
})

# Build and run pipeline
pipe = ScompLinkPipeline("Demo Numerical Prediction")
pipe.set_objectives(["Minimize RMSE"])
pipe.import_and_clean_data(df)
pipe.select_variables(target_col='y')
pipe.choose_model("numerical_prediction", 
                 metadata={"only_numerical_exogenous": True, 
                          "all_variables_important": False})
results = pipe.run_pipeline(task_type="regression")

print(results)
# Output: {'status': 'success', 'model_type': '...', 'metrics': {...}, 'report_path': '...'}
```

An HTML validation report is automatically generated: `ScompLink_Validation_Report.html`

---

## Complete Usage Guide

### Core Pipeline API

#### 1. Initialize Pipeline

```python
from scomp_link import ScompLinkPipeline

pipe = ScompLinkPipeline("Your Project Name")
```

#### 2. Set Objectives

```python
# For regression
pipe.set_objectives(["Minimize RMSE", "Maximize R2"])

# For classification
pipe.set_objectives(["Maximize Accuracy", "Maximize F1"])
```

#### 3. Import and Clean Data

```python
import pandas as pd

df = pd.read_csv("your_data.csv")
pipe.import_and_clean_data(df)
# Automatically removes duplicates and outliers
```

#### 4. Select Variables

```python
# Auto-select all features except target
pipe.select_variables(target_col='target_column')

# Or specify features manually
pipe.select_variables(target_col='target_column', 
                     feature_cols=['feature1', 'feature2'])
```

#### 5. Choose Model

The pipeline uses intelligent model selection based on your data characteristics:

```python
# Numerical Prediction
pipe.choose_model("numerical_prediction", 
                 metadata={
                     "only_numerical_exogenous": True,  # All features are numeric
                     "all_variables_important": False    # Feature selection needed
                 })

# Categorical Classification
pipe.choose_model("categorical_known", 
                 metadata={
                     "records_per_category": 500,
                     "exogenous_type": "mixed"  # categorical/numerical
                 })

# Clustering
pipe.choose_model("categorical_unknown", 
                 metadata={"categories_known": True})
```

#### 6. Run Pipeline

```python
# For regression
results = pipe.run_pipeline(task_type="regression", test_size=0.2)

# For classification
results = pipe.run_pipeline(task_type="classification", test_size=0.2)

# Access results
print(f"Model: {results['model_type']}")
print(f"Metrics: {results['metrics']}")
print(f"Report: {results['report_path']}")
```

---

## Advanced Usage

### Using Contrastive Learning for Text Classification (NEW! 🆕)

```python
from scomp_link.models.contrastive_text import ContrastiveTextClassifier
import pandas as pd

# Prepare data
df = pd.DataFrame({
    'text': ['AI revolutionizes tech', 'Team wins championship', ...],
    'category': ['Technology', 'Sports', ...]
})

# Initialize classifier
classifier = ContrastiveTextClassifier(
    model_name='bert-base-uncased',
    use_faiss=True,  # Fast inference
    embedding_dim=128
)

# Train with contrastive learning
classifier.train_contrastive(
    df,
    text_col='text',
    label_col='category',
    epochs=5,
    batch_size=64,
    validation_split=0.2
)

# Single prediction
prediction = classifier.predict("New smartphone with AI", top_k=3, return_confidence=True)
print(prediction)  # {'predictions': ['Technology', ...], 'confidences': [0.95, ...]}

# Batch prediction
test_df = pd.DataFrame({'text': test_texts})
results = classifier.predict_batch(test_df['text'], top_k=2)
print(results[['text', 'prediction', 'confidence']])

# Save/Load model
classifier.save('./models/my_classifier')
classifier.load('./models/my_classifier')
```

**Use Cases**:
- Text categorization with many classes
- Semantic similarity tasks
- Few-shot learning scenarios
- URL-to-App classification
- Document classification

**Advantages over traditional methods**:
- ✅ Better performance with many classes (100+)
- ✅ Works well with limited data per class
- ✅ Learns semantic relationships
- ✅ Fast inference with FAISS
- ✅ Transfer learning from BERT

---

### Using Optimizers Directly

#### Regression Optimizer

```python
from scomp_link.models.regressor_optimizer import RegressorOptimizer
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor

# Define models to test
models_to_test = {
    'LinearRegression': {
        'model': LinearRegression(),
        'params_grid': {
            'fit_intercept': [True, False]
        }
    },
    'Lasso': {
        'model': Lasso(),
        'params_grid': {
            'alpha': [0.1, 1.0, 10.0]
        }
    },
    'RandomForest': {
        'model': RandomForestRegressor(),
        'params_grid': {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None]
        }
    }
}

# Run optimizer
optimizer = RegressorOptimizer(
    df=df,
    y_col='target',
    x_cols=['feature1', 'feature2', 'feature3'],
    x_complexity_col='feature1',  # For visualization
    models_to_test=models_to_test,
    select_features=True  # Apply Boruta feature selection
)

# Estimate optimization time
optimizer.estimate_optimization_time(time_per_combination=60)

# Test all models
optimizer.test_models_regression()

# Access results
for model_name, results in optimizer.model_results.items():
    print(f"{model_name}: {results['Params']}")

# Generate visualization
fig = optimizer.grafico_fit_con_errore('LinearRegression')
fig.show()
```

#### Classification Optimizer

```python
from scomp_link.models.classifier_optimizer import ClassifierOptimizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

models_to_test = {
    'RandomForest': {
        'model': RandomForestClassifier(),
        'params_grid': {
            'n_estimators': [100, 200],
            'max_depth': [10, 20]
        }
    },
    'SVC': {
        'model': SVC(probability=True),
        'params_grid': {
            'C': [1, 10],
            'kernel': ['rbf', 'linear']
        }
    }
}

optimizer = ClassifierOptimizer(
    df=df,
    y_col='target',
    x_cols=['feature1', 'feature2'],
    models_to_test=models_to_test
)

optimizer.test_models_classification()
```

### Preprocessing Utilities

```python
from scomp_link import Preprocessor

# Initialize preprocessor
prep = Preprocessor(df)

# Clean data
cleaned_df = prep.clean_data(remove_outliers=True, outlier_threshold=3.0)

# Integrate external data
external_df = pd.read_csv("external_data.csv")
integrated_df = prep.integrate_data(external_df, on='id', how='left')

# Feature selection
top_features = prep.feature_selection(target_col='target', n_features=10)

# Run EDA
summary = prep.run_eda()
print(summary['shape'])
print(summary['missing_values'])

# Prepare train/test splits
X_train, X_test, y_train, y_test = prep.prepare_datasets('target', test_size=0.2)
```

### Validation and Metrics

```python
from scomp_link import Validator
from sklearn.linear_model import LinearRegression

# Train a model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Create validator
validator = Validator(model)

# Evaluate metrics
metrics = validator.evaluate(y_test, y_pred, task_type="regression")
print(f"RMSE: {metrics['rmse']:.4f}")
print(f"R²: {metrics['r2']:.4f}")

# K-Fold Cross Validation
cv_scores = validator.k_fold_cv(X_train, y_train, k=5)
print(f"CV Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Generate HTML report
validator.generate_validation_report(
    y_test, y_pred, 
    task_type="regression",
    report_name="My_Validation_Report.html"
)
```

### Custom HTML Reports

```python
from scomp_link.utils.report_html import ScompLinkHTMLReport
import plotly.express as px

# Create report
report = ScompLinkHTMLReport(
    title='Custom Analysis Report',
    main_color='#6E37FA',
    light_color='#9682FF',
    dark_color='#4614B4'
)

# Add sections
report.open_section("Data Analysis")
report.add_title("Distribution Analysis")
report.add_text("This section shows the distribution of key variables.")

# Add Plotly graphs
fig = px.scatter(df, x='x1', y='y', title='Scatter Plot')
report.add_graph_to_report(fig, 'Feature vs Target')

# Add dataframes
report.add_dataframe(df.head(20), 'Sample Data')

report.close_section()

# Save report
report.save_html('custom_report.html')
```

### Visualization Utilities

```python
from scomp_link.utils.plotly_utils import (
    histogram, multiple_histograms, 
    barchart, linechart, area_chart
)

# Single histogram
fig = histogram(df['age'], 'Age Distribution', h=600)
fig.show()

# Multiple histograms by category
fig = multiple_histograms(
    df['value'], 
    df['category'],
    category_name='Product Category',
    y_label='Sales',
    h=300
)
fig.show()

# Bar chart
fig = barchart(
    categories=['A', 'B', 'C'],
    metric_values_list=[[10, 20, 30], [15, 25, 35]],
    y_axis_titles=['Metric 1', 'Metric 2']
)
fig.show()

# Line chart
fig = linechart(
    date_list=['2024-01-01', '2024-01-02', '2024-01-03'],
    lines=[[10, 15, 20], [5, 10, 15]],
    y_labels=['Series 1', 'Series 2'],
    title_text='Time Series Analysis'
)
fig.show()
```

---

## Model Selection Decision Tree

The pipeline automatically selects the best model based on your data:

### Numerical Prediction
- **< 1000 records**: Econometric Model
- **1000-100k records**:
  - Only numerical features:
    - All important: Ridge / SVR
    - Feature selection needed: Lasso / Elastic Net
  - Mixed features: Gradient Boosting / Random Forest
- **> 100k records**:
  - Only numerical: SGD Regressor
  - Mixed: Gradient Boosting / Random Forest

### Categorical Classification
- **Image data**:
  - < 500 per category: Pre-trained model
  - ≥ 500 per category: CNN (ResNet/Inception)
- **Categorical features**:
  - < 5 features: Theorical Psychometric Model
  - ≥ 5 features: Naive Bayes / Classification Tree
- **Mixed features**:
  - < 300 per category: SVC / K-Neighbors / Naive Bayes
  - ≥ 300 per category: SGD / Gradient Boosting / Random Forest

### Clustering
- Categories known: KMeans / Hierarchical Clustering
- Categories unknown: Mean-Shift Clustering

---

## Validation Reports

Every pipeline run generates an HTML report containing:

### Regression Reports
- **Metrics Summary**: MSE, RMSE, MAE, R²
- **Observed vs Predicted**: Scatter plot with ideal line
- **Residuals Distribution**: Histogram of prediction errors
- **Residuals Analysis**: Binned residuals with confidence intervals

### Classification Reports
- **Metrics Summary**: Accuracy, F1, Precision, Recall
- **Confusion Matrix**: Interactive heatmap
- **Confidence Distribution**: Probability distributions per class
- **ROC Curves**: (when applicable)

All reports are:
- ✅ Self-contained HTML files
- ✅ Interactive (Plotly-based)
- ✅ Responsive design
- ✅ Exportable to CSV

---

## Ensemble Learning & Advanced Cross-Validation (NEW! 🆕)

### Ensemble Learning

Combine multiple models for improved performance:

```python
from scomp_link import ScompLinkPipeline

# Define multiple models to test
models_to_test = {
    'Ridge': {'model': Ridge(), 'params_grid': {'alpha': [0.1, 1.0, 10.0]}},
    'Lasso': {'model': Lasso(), 'params_grid': {'alpha': [0.1, 1.0, 10.0]}},
    'RandomForest': {'model': RandomForestRegressor(), 'params_grid': {'n_estimators': [50, 100]}}
}

pipe = ScompLinkPipeline("Ensemble Demo")
pipe.import_and_clean_data(df)
pipe.select_variables(target_col='y')

# Run with ensemble
results = pipe.run_pipeline(
    task_type="regression",
    models_to_test=models_to_test,
    use_ensemble=True,              # Enable ensemble
    ensemble_strategy='voting'       # or 'stacking'
)

print(f"Ensemble Score: {results['ensemble_scores']['mean_score']:.4f}")
```

**Strategies:**
- **Voting**: Average predictions from all models
- **Stacking**: Use meta-learner to combine predictions

### Advanced Cross-Validation

Go beyond K-Fold with LOOCV and Bootstrap:

```python
results = pipe.run_pipeline(
    task_type="regression",
    models_to_test=models_to_test,
    advanced_cv=True,                    # Enable advanced CV
    cv_methods=['loocv', 'bootstrap'],   # Validation methods
    bootstrap_iterations=1000            # Bootstrap samples
)

# Access advanced CV results
for method, cv_result in results['advanced_cv'].items():
    print(f"{cv_result['method']}: {cv_result['mean_score']:.4f}")
    if 'confidence_interval_95' in cv_result:
        ci = cv_result['confidence_interval_95']
        print(f"  95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
```

**Methods:**
- **LOOCV (C1)**: Leave-One-Out Cross Validation (for small datasets)
- **Bootstrap (C3)**: Resampling with confidence intervals
- **K-Fold (C2)**: Standard cross-validation (default)

See [Ensemble & Advanced CV Documentation](scomp_link/models/README_ENSEMBLE.md) for details.

---

## Testing

The package includes comprehensive tests with ~100% coverage:

```bash
# Run all tests
python3 -m pytest tests/test_comprehensive.py -v

# Run with coverage report
python3 -m pytest tests/test_comprehensive.py --cov=scomp_link --cov-report=html

# Run specific test class
python3 -m pytest tests/test_comprehensive.py::TestScompLinkPipeline -v
```

Test coverage includes:
- ✅ Core pipeline functionality (12 tests)
- ✅ Preprocessing operations (8 tests)
- ✅ Model factory (9 tests)
- ✅ Validation and metrics (6 tests)
- ✅ Integration workflows (3 tests)
- ✅ Edge cases (3 tests)

---

## Project Structure

```
scomp_link/
├── scomp_link/              # Main package
│   ├── core.py              # ScompLinkPipeline orchestrator
│   ├── preprocessing/       # Data cleaning and preparation
│   │   └── data_processor.py
│   ├── models/              # Model implementations
│   │   ├── model_factory.py
│   │   ├── regressor_optimizer.py
│   │   ├── classifier_optimizer.py
│   │   ├── supervised_text.py
│   │   ├── supervised_img.py
│   │   ├── unsupervised_text.py
│   │   ├── unsupervised_img.py
│   │   ├── contrastive_net.py
│   │   └── url_to_app_model.py
│   ├── validation/          # Model validation
│   │   ├── model_validator.py
│   │   └── validation_model.py
│   └── utils/               # Utilities
│       ├── report_html.py
│       └── plotly_utils.py
├── tests/                   # Test suite
│   └── test_comprehensive.py
├── requirements.txt         # Core dependencies
├── setup.py                 # Package configuration
└── README.md                # This file
```

---

## Design Principles

1. **Generalized**: No project-specific behavior or assumptions
2. **Pluggable**: Optional dependencies loaded on-demand
3. **Consistent APIs**: Unified interfaces across all models and tools
4. **Automation-First**: Minimize manual configuration while maintaining flexibility
5. **Fail Gracefully**: Optional features degrade without breaking core functionality

---

## Dependencies

### Core (Always Required)
- numpy
- pandas
- scipy
- scikit-learn
- matplotlib
- plotly
- boruta

### Optional
- **NLP**: torch, transformers, spacy
- **Computer Vision**: tensorflow, pillow
- **Utilities**: tqdm, PyJWT

---

## Contributing

Contributions are welcome! Please ensure:
- All tests pass (`pytest tests/`)
- Code follows existing patterns
- Documentation is updated
- New features include tests

---

## License

MIT License - See repository-level license file.

---

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check existing documentation
- Review test examples in `tests/test_comprehensive.py`

---

**May the code be with you.** 🚀
