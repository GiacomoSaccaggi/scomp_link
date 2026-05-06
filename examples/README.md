# Examples Documentation

## Overview

This directory contains 18 comprehensive examples demonstrating all features of scomp-link, organized by workflow phase and use case.

## Examples by Workflow Phase

### Preprocessing & Regression (P1-P12)

**example_01_numerical_small.py**
- Small dataset (< 1000 records)
- Automatic preprocessing
- Econometric model selection
- **Workflow:** P3-P4 → Model Selection → Validation

**example_02_numerical_medium_lasso.py**
- Medium dataset (1k-100k records)
- Feature selection with Lasso
- **Workflow:** P10 Feature Selection → Lasso/Elastic Net

**example_03_numerical_mixed_features.py**
- Mixed categorical and numerical features
- Automatic encoding and scaling
- **Workflow:** P7 Transformation → Gradient Boosting

### Classification

**example_04_classification_small.py**
- Small classification dataset
- < 300 records per category
- **Workflow:** SVC / K-Neighbors / Naive Bayes

**example_05_classification_large.py**
- Large classification dataset
- ≥ 300 records per category
- **Workflow:** SGD / Gradient Boosting / Random Forest

### Clustering

**example_06_clustering_known.py**
- Known number of clusters
- **Workflow:** KMeans / Hierarchical Clustering

**example_07_clustering_unknown.py**
- Unknown number of clusters
- **Workflow:** Mean-Shift Clustering

### Large Dataset

**example_08_numerical_very_large.py**
- > 100,000 records
- **Workflow:** SGD Regressor for scalability

### Text Classification

**example_09_text_classification.py**
- Contrastive learning text classification
- BERT-based embeddings + FAISS
- Save/load model and predict
- **Workflow:** Contrastive training → Semantic similarity → Prediction

**example_12_text_configuration.py**
- Configuration options comparison
- Option 1: Contrastive Learning (BERT + FAISS)
- Option 2: TF-IDF + SGD (fast, simple)
- Save/load and predict demo

**example_13_text_unsupervised.py**
- Text clustering with sentence-transformers
- Semantic embeddings + KMeans
- Save/load clusterer

### Image

**example_10_image_classification.py**
- Image classification with CNN
- < 500 images per category → Pre-trained
- ≥ 500 images per category → CNN training

**example_11_image_clustering.py**
- Unsupervised image clustering
- Feature extraction + KMeans

### Ensemble & Advanced Validation

**example_14_ensemble_advanced_cv.py**
- Ensemble Learning (Voting/Stacking)
- Advanced Cross-Validation (LOOCV/Bootstrap)
- Multiple models comparison
- **Workflow:** Multiple models → Ensemble → Advanced validation

### Anomaly Detection

**example_15_anomaly_detection.py**
- Multi-method anomaly detection for tabular data
- 4 methods: Isolation Forest, LOF, TabNet Autoencoder, Transformer Autoencoder
- Consensus voting for robust anomaly labels
- **Workflow:** Data → Multi-method detection → Consensus → Report

**example_16_ts_anomaly_detection.py**
- Time series anomaly detection
- 4 methods: Conv1D Autoencoder, Moving Average, Moving Median, ARIMA Residuals
- Train on normal data, detect on new data
- **Workflow:** Fit normal → Detect anomalies → Per-method results

### Reporting

**example_17_report_html.py**
- Full HTML report generation demo
- Titles, text, DataFrames, collapsible sections
- Plotly graphs (single, combobox selection, multi-filter)
- Matplotlib graphs (base64 embedded)
- Plotly utilities: histogram, barchart, linechart, area_chart
- PDF export (requires playwright)

### Contrastive Learning

**example_18_contrastive_text_example.py**
- Direct usage of ContrastiveTextClassifier
- BERT-based contrastive embeddings
- Few-shot learning scenario
- Save/load model
- **Workflow:** Contrastive training → Semantic similarity → Prediction

## Running Examples

### Run All Examples (via tox, multi-version)
```bash
rm -rf .tox && ./run_tox_all_versions.sh
```

### Run All Examples (current env)
```bash
bash run_all_examples.sh
```

### Run Individual Example
```bash
python3 examples/example_01_numerical_small.py
```

## Example Structure

Each example follows this structure:

```python
# 1. Import libraries
from scomp_link import ScompLinkPipeline
import pandas as pd
import numpy as np

# 2. Generate/Load data
df = pd.DataFrame({...})

# 3. Initialize pipeline
pipe = ScompLinkPipeline("Example Name")

# 4. Set objectives
pipe.set_objectives(["Objective 1", "Objective 2"])

# 5. Preprocessing
pipe.import_and_clean_data(df)
pipe.select_variables(target_col='y')

# 6. Model selection
pipe.choose_model("objective_type", metadata={...})

# 7. Run pipeline
results = pipe.run_pipeline(
    task_type="regression",
    # ... additional parameters
)

# 8. Display results
print(results)
```

## Workflow Mapping

| Example | Workflow Phases | Decision Tree Branch |
|---------|----------------|---------------------|
| 01 | P3-P4 → < 1k records | Econometric Model |
| 02 | P10 → 1k-100k, feature selection | Lasso/Elastic Net |
| 03 | P7 → 1k-100k, mixed features | Gradient Boosting |
| 04 | Classification, < 300/category | SVC/K-Neighbors |
| 05 | Classification, ≥ 300/category | SGD/GB/RF |
| 06 | Clustering, known categories | KMeans |
| 07 | Clustering, unknown categories | Mean-Shift |
| 08 | > 100k records | SGD Regressor |
| 09 | Text, contrastive learning | BERT + FAISS |
| 10 | Images, classification | CNN/Pre-trained |
| 11 | Images, clustering | Feature extraction |
| 12 | Text, configuration options | Contrastive / TF-IDF |
| 13 | Text, unsupervised | Sentence-transformers + KMeans |
| 14 | Ensemble + Advanced CV | Multiple models |
| 15 | Anomaly detection, tabular | IForest/LOF/TabNet/Transformer |
| 16 | Anomaly detection, time series | AE/MovAvg/MovMed/ARIMA |
| 17 | HTML/PDF report generation | ScompLinkHTMLReport |
| 18 | Text, contrastive (direct API) | ContrastiveTextClassifier |

## Examples by Feature

### Basic Features
- **Data Cleaning**: examples 01-03
- **Feature Selection**: example 02
- **Model Selection**: all examples
- **Validation**: all examples

### Advanced Features
- **Ensemble Learning**: example 14
- **Advanced CV**: example 14
- **Anomaly Detection (Tabular)**: example 15
- **Anomaly Detection (Time Series)**: example 16
- **Text Classification**: examples 09, 12, 18
- **Text Clustering**: example 13
- **Image Processing**: examples 10, 11
- **Clustering**: examples 06, 07, 11, 13
- **HTML Reporting**: example 17

### Optimization Features
- **Grid Search**: examples 02, 03, 05
- **Hyperparameter Tuning**: examples 02-05
- **Cross-Validation**: all examples
- **Bootstrap**: example 14
- **LOOCV**: example 14

## Troubleshooting

### Common Issues

**1. Missing Dependencies**
```bash
pip install scomp-link
```

**2. Memory Issues (Large Datasets)**
```python
# Reduce dataset size
df = df.sample(n=10000, random_state=42)
```

**3. Slow Optimization**
```python
# Reduce parameter grid
models_to_test = {
    'Model': {
        'model': Model(),
        'params_grid': {'param': [value1, value2]}
    }
}
```

**4. PDF Export (example 17)**
```bash
pip install playwright
playwright install chromium
```

## See Also

- [Main README](../README.md) - API reference
- [Workflow Documentation](../WORKFLOW.md) - Complete workflow mapping
- [Ensemble Learning](../scomp_link/models/README_ENSEMBLE.md) - Advanced features
- [VERSION_REQUIREMENTS.md](../VERSION_REQUIREMENTS.md) - Dependency versions
