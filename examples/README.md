# Examples Documentation

## Overview

This directory contains 14 comprehensive examples demonstrating all features of scomp-link, organized by workflow phase and use case.

## Examples by Workflow Phase

### Preprocessing Examples (P1-P12)

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

### Classification Examples

**example_04_classification_small.py**
- Small classification dataset
- < 300 records per category
- **Workflow:** SVC / K-Neighbors / Naive Bayes

**example_05_classification_large.py**
- Large classification dataset
- ≥ 300 records per category
- **Workflow:** SGD / Gradient Boosting / Random Forest

### Clustering Examples

**example_06_clustering_known.py**
- Known number of clusters
- **Workflow:** KMeans / Hierarchical Clustering

**example_07_clustering_unknown.py**
- Unknown number of clusters
- **Workflow:** Mean-Shift Clustering

### Large Dataset Examples

**example_08_numerical_very_large.py**
- > 100,000 records
- **Workflow:** SGD Regressor for scalability

### Text Classification Examples

**example_09_text_classification.py**
- Traditional text classification
- Spacy + CNN
- **Workflow:** Text preprocessing → CNN training

**example_12_text_configuration.py**
- Advanced text configuration
- Custom language and model selection

**example_13_text_unsupervised.py**
- Text clustering
- Topic modeling

### Image Examples

**example_10_image_classification.py**
- Image classification with CNN
- < 500 images per category → Pre-trained
- ≥ 500 images per category → CNN training

**example_11_image_clustering.py**
- Unsupervised image clustering
- Feature extraction + KMeans

### Advanced Features (NEW!)

**example_14_ensemble_advanced_cv.py**
- Ensemble Learning (Voting/Stacking)
- Advanced Cross-Validation (LOOCV/Bootstrap)
- Complete workflow demonstration
- **Workflow:** Multiple models → Ensemble → Advanced validation

**contrastive_text_example.py**
- Contrastive learning for text
- BERT-based embeddings
- Few-shot learning
- **Workflow:** Contrastive training → Semantic similarity

## Running Examples

### Run All Examples
```bash
bash run_all_examples.sh
```

### Run Individual Example
```bash
python3 examples/example_01_numerical_small.py
```

### Run with Virtual Environment
```bash
# Create environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -e .[all]

# Run example
python3 examples/example_14_ensemble_advanced_cv.py
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

## Examples by Feature

### Basic Features
- **Data Cleaning**: examples 01-03
- **Feature Selection**: example 02
- **Model Selection**: all examples
- **Validation**: all examples

### Advanced Features
- **Ensemble Learning**: example 14
- **Advanced CV**: example 14
- **Text Classification**: examples 09, 12, 13, contrastive
- **Image Processing**: examples 10, 11
- **Clustering**: examples 06, 07, 11, 13

### Optimization Features
- **Grid Search**: examples 02, 03, 05
- **Hyperparameter Tuning**: examples 02-05
- **Cross-Validation**: all examples
- **Bootstrap**: example 14
- **LOOCV**: example 14

## Expected Outputs

Each example generates:

1. **Console Output**
   - Progress messages
   - Model selection decisions
   - Training progress
   - Metrics summary

2. **HTML Report**
   - `ScompLink_Validation_Report.html`
   - Interactive visualizations
   - Metrics tables
   - Model diagnostics

3. **Return Dictionary**
   ```python
   {
       "status": "success",
       "model_type": "...",
       "metrics": {...},
       "advanced_cv": {...},  # If enabled
       "ensemble_scores": {...},  # If enabled
       "report_path": "..."
   }
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
| 09 | Text, supervised | CNN + Spacy |
| 10 | Images, classification | CNN/Pre-trained |
| 11 | Images, clustering | Feature extraction |
| 12 | Text, configuration | Custom NLP |
| 13 | Text, unsupervised | Topic modeling |
| 14 | Ensemble + Advanced CV | Multiple models |
| contrastive | Text, contrastive learning | BERT embeddings |

## Customization

### Modify Example Parameters

```python
# Change dataset size
N = 5000  # Instead of 1000

# Change model parameters
models_to_test = {
    'CustomModel': {
        'model': YourModel(),
        'params_grid': {
            'param1': [value1, value2],
            'param2': [value3, value4]
        }
    }
}

# Change validation strategy
results = pipe.run_pipeline(
    task_type="regression",
    test_size=0.3,  # Instead of 0.2
    advanced_cv=True,
    cv_methods=['loocv', 'bootstrap'],
    bootstrap_iterations=2000  # Instead of 1000
)
```

### Add Custom Preprocessing

```python
from scomp_link.preprocessing import Preprocessor

# Custom preprocessing
prep = Preprocessor(df)
df_clean = prep.clean_data(outlier_threshold=2.5)
df_integrated = prep.integrate_data(external_df, on='id')
selected_features = prep.feature_selection(target_col='y', n_features=15)

# Use in pipeline
pipe.import_and_clean_data(df_integrated)
pipe.select_variables(target_col='y', feature_cols=selected_features)
```

## Troubleshooting

### Common Issues

**1. Missing Dependencies**
```bash
# Install all dependencies
pip install -e .[all]

# Or specific groups
pip install -e .[nlp]  # For text examples
pip install -e .[img]  # For image examples
```

**2. Memory Issues (Large Datasets)**
```python
# Reduce dataset size
df = df.sample(n=10000, random_state=42)

# Or use SGD for large data
pipe.choose_model("numerical_prediction", metadata={
    "only_numerical_exogenous": True
})
# Automatically selects SGD for > 100k records
```

**3. Slow Optimization**
```python
# Reduce parameter grid
models_to_test = {
    'Model': {
        'model': Model(),
        'params_grid': {
            'param': [value1, value2]  # Fewer values
        }
    }
}

# Or estimate time first
optimizer.estimate_optimization_time(time_per_combination=60)
```

**4. LOOCV Too Slow**
```python
# Use bootstrap instead
results = pipe.run_pipeline(
    advanced_cv=True,
    cv_methods=['bootstrap'],  # Skip LOOCV
    bootstrap_iterations=500
)
```

## Performance Tips

1. **Start Small**: Test with subset of data first
2. **Estimate Time**: Use `estimate_optimization_time()`
3. **Parallel Processing**: Enable in optimizers (future feature)
4. **Cache Results**: Save trained models with `pipe.save_model()`
5. **Skip LOOCV**: For datasets > 1000 samples

## Next Steps

After running examples:

1. **Explore Reports**: Open generated HTML files
2. **Modify Parameters**: Experiment with different settings
3. **Try Your Data**: Replace synthetic data with real datasets
4. **Combine Features**: Use ensemble + advanced CV together
5. **Read Documentation**: Check module READMEs for details

## See Also

- [Main README](../README.md) - API reference
- [Workflow Documentation](../WORKFLOW.md) - Complete workflow mapping
- [Preprocessing](../scomp_link/preprocessing/README.md) - P1-P12 phases
- [Models](../scomp_link/models/README.md) - Model selection
- [Validation](../scomp_link/validation/README.md) - Validation strategies
- [Ensemble Learning](../scomp_link/models/README_ENSEMBLE.md) - Advanced features
