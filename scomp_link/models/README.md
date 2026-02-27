# Models Module

## Overview

The models module implements the intelligent model selection and optimization logic based on the complete decision tree workflow. It automatically selects the appropriate algorithm based on data characteristics, objective type, and dataset size.

## Decision Tree Implementation

This module implements the complete "Tipo di obbiettivo" (Objective Type) decision tree:

### 1. Categorical Known (Classification)

**Images:**
- < 500 per category → Pre-trained model
- ≥ 500 per category → CNN (ResNet/Inception)

**Categorical Variables:**
- Few variables → Theoretical/Psychometric Model
- Many variables → Naive Bayes / Classification Tree

**Mixed Variables (Categorical + Numerical):**
- < 300 records/category → SVC / K-Neighbors / Naive Bayes
- ≥ 300 records/category → SGD / Gradient Boosting / Random Forest

### 2. Categorical Unknown (Clustering)

- Categories unknown → Mean-Shift Clustering
- Categories known → KMeans / Hierarchical Clustering

### 3. Numerical Study

- Geospatial data → Geostatistical Model / Kriging
- Time series → UCM State Space
- Other → Randomized PCA / Statistical Tests

### 4. Numerical Prediction (Regression)

**< 1,000 records:**
- Econometric Model

**1,000 - 100,000 records:**
- Only numerical variables:
  - All important → Ridge / SVR
  - Some unimportant → Lasso / Elastic Net
- Mixed variables → Gradient Boosting / Random Forest

**> 100,000 records:**
- Only numerical → SGD Regressor
- Mixed variables → Gradient Boosting / Random Forest

### 5. Multi-Numerical Prediction

- Time series → VAR / VARMA
- Other → Multilayer Perceptron (MLP)

## Core Components

### ModelFactory

Intelligent model selection based on decision tree:

```python
from scomp_link.models import ModelFactory

# Automatic model selection
model = ModelFactory.get_model(
    "Ridge / SVR",
    alpha=1.0
)

# Based on metadata
model = ModelFactory.get_model(
    model_type="auto",
    metadata={
        "objective": "numerical_prediction",
        "n_records": 5000,
        "only_numerical": True,
        "all_important": False
    }
)
```

### RegressorOptimizer

Grid search optimization for regression models:

```python
from scomp_link.models import RegressorOptimizer
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor

# Define models to test
models_to_test = {
    'Ridge': {
        'model': Ridge(),
        'params_grid': {
            'alpha': [0.1, 1.0, 10.0]
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
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, None]
        }
    }
}

# Optimize
optimizer = RegressorOptimizer(
    df=df,
    y_col='target',
    x_cols=['feature1', 'feature2', 'feature3'],
    x_complexity_col='feature1',
    models_to_test=models_to_test,
    select_features=True  # Automatic feature selection
)

# Estimate time
optimizer.estimate_optimization_time(time_per_combination=60)

# Test all models
optimizer.test_models_regression()

# Access results
for model_name, results in optimizer.model_results.items():
    print(f"{model_name}: {results['Params']}")

# Visualize fit
fig = optimizer.grafico_fit_con_errore('Ridge')
```

### ClassifierOptimizer

Grid search optimization for classification models:

```python
from scomp_link.models import ClassifierOptimizer
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

### EnsembleOptimizer

Combine multiple models for improved performance:

```python
from scomp_link.models import EnsembleOptimizer

# After training multiple models
base_models = [
    ('ridge', ridge_model),
    ('lasso', lasso_model),
    ('rf', rf_model)
]

# Voting ensemble
ensemble = EnsembleOptimizer(
    base_models=base_models,
    task_type='regression',
    strategy='voting'
)
ensemble.fit(X_train, y_train)
predictions = ensemble.predict(X_test)

# Stacking ensemble
ensemble_stack = EnsembleOptimizer(
    base_models=base_models,
    task_type='regression',
    strategy='stacking'
)
ensemble_stack.create_ensemble(meta_model=Ridge())
ensemble_stack.fit(X_train, y_train)

# Evaluate
scores = ensemble.evaluate_ensemble(X, y, cv=5)
print(f"Mean CV Score: {scores['mean_score']:.4f}")
```

## Specialized Models

### Text Classification

```python
from scomp_link.models.supervised_text import SpacyEmbeddingModel

# Traditional approach
classifier = SpacyEmbeddingModel(lan='en', model_name='bert-base-uncased')
nlp_model, scores = classifier.training(to_tag, tagged, categories)

# Contrastive learning approach
from scomp_link.models.contrastive_text import ContrastiveTextClassifier

classifier = ContrastiveTextClassifier(
    model_name='bert-base-uncased',
    embedding_dim=128,
    use_faiss=True
)
classifier.train_contrastive(df, text_col='text', label_col='category')
```

### Image Classification

```python
from scomp_link.models.supervised_img import CNNImg

# CNN for images
cnn = CNNImg(input_shape=(224, 224, 3), n_classes=10)
cnn.fit(X_train, y_train)
predictions = cnn.predict(X_test)
```

### Image Clustering

```python
from scomp_link.models.unsupervised_img import ClusterImg

clusterer = ClusterImg(n_clusters=5, method='kmeans')
clusters = clusterer.fit_predict(X_images)
```

## Model Selection Logic

The module automatically selects models based on:

1. **Dataset size**: < 1k, 1k-100k, > 100k records
2. **Feature types**: numerical, categorical, mixed, images, text
3. **Task type**: regression, classification, clustering
4. **Records per category**: For classification tasks
5. **Feature importance**: All important vs. some unimportant

## Hyperparameter Optimization (M3)

All optimizers implement:

- **Grid Search**: Exhaustive search over parameter grid
- **Cross-Validation**: K-Fold CV for robust evaluation
- **Best Model Selection**: Automatic selection of best parameters
- **Time Estimation**: Predict optimization duration

## Integration with Pipeline

```python
from scomp_link import ScompLinkPipeline

pipe = ScompLinkPipeline("Project")
pipe.import_and_clean_data(df)
pipe.select_variables(target_col='y')

# Automatic model selection
pipe.choose_model(
    objective_type="numerical_prediction",
    metadata={
        "only_numerical_exogenous": True,
        "all_variables_important": False
    }
)

# Run with multiple models + ensemble
results = pipe.run_pipeline(
    task_type="regression",
    models_to_test=models_dict,
    use_ensemble=True,
    ensemble_strategy='voting'
)
```

## Model Validation Parameters (M4)

Supported cross-validation strategies:

- **K-Fold Cross Validation** (C2)
- **Leave-One-Out Cross Validation (LOOCV)** (C1)
- **Bootstrap** (parametric/non-parametric) (C3)
- **Epoch selection** for neural networks (C4)

See [Advanced CV Documentation](README_ENSEMBLE.md) for details.

## Best Practices

1. **Start with multiple models**: Test several algorithms
2. **Use feature selection**: Enable `select_features=True`
3. **Estimate time first**: Call `estimate_optimization_time()`
4. **Consider ensemble**: Use voting/stacking for better performance
5. **Validate properly**: Use appropriate CV strategy for dataset size

## Dependencies

- scikit-learn: Core ML algorithms
- numpy, pandas: Data manipulation
- torch, transformers: Deep learning (optional)
- tensorflow: Image models (optional)
- spacy: NLP (optional)

## See Also

- [Preprocessing](../preprocessing/README.md)
- [Validation](../validation/README.md)
- [Ensemble Learning](README_ENSEMBLE.md)
