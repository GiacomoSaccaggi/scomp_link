# scomp-link API Reference

Complete English documentation of all classes, methods, and functions in the `scomp_link` package.

---

## Table of Contents

1. [Core — `scomp_link.core`](#core)
2. [Preprocessing — `scomp_link.preprocessing.data_processor`](#preprocessing)
3. [Models](#models)
   - [Model Factory](#model-factory)
   - [Regressor Optimizer](#regressor-optimizer)
   - [Boruta Feature Selection](#boruta-feature-selection)
   - [Classifier Optimizer](#classifier-optimizer)
   - [Ensemble Optimizer](#ensemble-optimizer)
   - [Anomaly Detector (Tabular)](#anomaly-detector-tabular)
   - [Time Series Anomaly Detector](#time-series-anomaly-detector)
   - [Contrastive Text Classifier](#contrastive-text-classifier)
   - [Contrastive Network Components](#contrastive-network-components)
   - [Supervised Text (SpacyEmbeddingModel)](#supervised-text)
   - [Supervised Image (CNNImg)](#supervised-image)
   - [Unsupervised Text (TextEmbeddingClustering)](#unsupervised-text)
   - [Unsupervised Image (ClusterImg)](#unsupervised-image)
4. [Validation](#validation)
   - [Validator](#validator)
   - [Advanced Cross-Validation](#advanced-cross-validation)
5. [Utilities](#utilities)
   - [HTML Report Builder](#html-report-builder)
   - [Plotly Utilities](#plotly-utilities)
   - [Highcharts Utilities](#highcharts-utilities)
   - [PDF Converter](#pdf-converter)

---

## Core

### `class ScompLinkPipeline`

**Module:** `scomp_link.core`

The main orchestrator class — the "Astromech arm" for Python data projects. Automates the full ML workflow from problem formulation to model evaluation.

#### `__init__(self, problem_description: str)`

Initialize the pipeline with a human-readable description of the problem being solved. Sets up internal state for objectives, data, target, features, model, and results.

#### `set_objectives(self, objectives: List[str])`

Define the analysis objectives (e.g., "Minimize RMSE", "Maximize Accuracy"). Corresponds to the Objectives Formulation phase of the workflow.

#### `import_and_clean_data(self, df: pd.DataFrame)`

Import a DataFrame and perform automatic cleaning (P3–P4 phases). Creates a `Preprocessor` instance, removes duplicates and outliers. Stores the cleaned DataFrame internally.

#### `select_variables(self, target_col: str, feature_cols: Optional[List[str]] = None)`

Select the target variable and feature columns for modeling. If `feature_cols` is not provided, all columns except the target are used as features.

#### `choose_model(self, objective_type: str, metadata: Optional[Dict[str, Any]] = None)`

Select a model based on the decision-tree logic. Uses `objective_type` to navigate the decision tree and `metadata` to determine the specific algorithm.

**Supported objective types:**
- `"categorical_known"` — Classification with known categories (images, categorical, or mixed features)
- `"categorical_unknown"` — Clustering (KMeans/Hierarchical if categories known, Mean-Shift otherwise)
- `"numerical_study"` — Numerical analysis (geostatistical, time series, or PCA/statistical tests)
- `"numerical_prediction"` — Regression (model chosen based on dataset size and feature types)
- `"multi_numerical_prediction"` — Multi-target prediction (VAR/VARMA for time series, MLP otherwise)

#### `run_pipeline(self, test_size=0.2, task_type="regression", models_to_test=None, text_col=None, image_col=None, n_clusters=None, epochs=3, batch_size=32, text_model='bert-base-uncased', text_language='en', use_contrastive=True, use_ensemble=False, ensemble_strategy='voting', advanced_cv=False, cv_methods=['bootstrap'], bootstrap_iterations=1000)`

Execute the full modeling pipeline: split data, train model(s), evaluate, and generate validation report.

**Supported task types:**
- `"regression"` — Regression with optional multi-model optimization and ensemble
- `"classification"` — Classification with optional multi-model optimization and ensemble
- `"clustering"` — KMeans or Mean-Shift clustering
- `"text"` — Text classification (contrastive BERT or TF-IDF + SGD)
- `"image"` — Image classification (CNN)
- `"image_clustering"` — Image feature clustering

**Returns:** Dictionary with status, model type, metrics, report path, and optional ensemble/advanced CV results.

#### `save_model(self, path='./staging')`

Save the trained model to disk. Uses the model's native `save()` method for contrastive text models, or `pickle` for standard sklearn models.

#### `load_model(self, path='./staging')`

Load a previously saved model from disk. Automatically detects whether it's a contrastive text model (directory with `metadata.json`) or a standard pickled sklearn model.

#### `predict(self, X)`

Make predictions using the loaded model. Handles both standard sklearn models (array input) and ContrastiveTextClassifier (list of strings input).

---

## Preprocessing

### `class Preprocessor`

**Module:** `scomp_link.preprocessing.data_processor`

Handles preprocessing phases P1–P12 as described in the scomp-link workflow schema.

#### `__init__(self, df: pd.DataFrame)`

Initialize with a DataFrame. Stores both a working copy and the original for reference.

#### `clean_data(self, remove_outliers: bool = True, outlier_threshold: float = 3.0)`

**P4: Data Cleaning** — Removes formal/logical errors and outliers. Drops duplicate rows and removes records where any numeric column has a Z-score above the threshold.

#### `integrate_data(self, other_df: pd.DataFrame, on: str, how: str = 'left')`

**P5: Data Integration (Record Linkage)** — Combines data from different sources by merging DataFrames on a shared key column.

#### `transform_data(self)`

**P7: Data Transformation** — Manipulates data into forms suitable for data mining. Currently a placeholder for scaling, encoding, and other transformations.

#### `feature_selection(self, target_col: str, n_features: Optional[int] = None)`

**P10: Feature Selection** — Selects the most important features using correlation-based ranking against the target variable. Returns the top N features sorted by absolute correlation.

#### `run_eda(self)`

**P11: Exploratory Data Analysis** — Generates a summary dictionary containing dataset shape, missing value counts, data types, and descriptive statistics.

#### `prepare_datasets(self, target_col: str, test_size: float = 0.2)`

**P12: Dataset Preparation** — Splits the DataFrame into training and test sets (X_train, X_test, y_train, y_test) using `random_state=42` for reproducibility.

---

## Models

### Model Factory

### `class ModelFactory`

**Module:** `scomp_link.models.model_factory`

Factory class that creates model instances based on the scomp-link decision tree string identifiers.

#### `@staticmethod get_model(model_type: str, **kwargs)`

Create and return a model instance based on a string identifier. Uses substring matching to map model type strings to concrete sklearn/custom model classes.

**Supported model types:**
- Optimizers: `"ClassifierOptimizer"`, `"RegressorOptimizer"`
- Text: `"Contrastive Text"`, `"Spacy"`, `"Supervised Text"`, `"LDA"`, `"Unsupervised Text"`
- Image: `"CNN"`, `"Supervised Img"`, `"Cluster Img"`, `"Unsupervised Img"`
- Classification: `"Naive Bayes"`, `"Classification Tree"`, `"SVC"`, `"K-Neighbors"`, `"SGD Classifier"`, `"Gradient Boosting Classifier"`, `"Random Forest Classifier"`
- Regression: `"Ridge"`, `"Lasso"`, `"Elastic Net"`, `"SVR"`, `"SGD Regressor"`, `"Gradient Boosting"`, `"Random Forest"`
- Clustering: `"KMeans"`, `"Mean-Shift"`
- Other: `"Econometric Model"`, `"Theorical Psychometric Model"` (both map to LinearRegression)

---

### Regressor Optimizer

### `class RegressorOptimizer`

**Module:** `scomp_link.models.regressor_optimizer`

Optimizes regression models by performing feature selection (Boruta), hyperparameter tuning (GridSearchCV), and evaluating multiple regression algorithms on the same dataset.

#### `__init__(self, df, y_col, x_cols, x_complexity_col, models_to_test, select_features=False)`

Initialize the optimizer. Splits the dataset into features (X) and target (y). Identifies categorical, binary, and numeric columns and creates a `ColumnTransformer` with appropriate preprocessing (OneHotEncoder for categorical/binary, StandardScaler for numeric). Splits data into train/test (80/20) and sets up 5-fold cross-validation.

**Parameters:**
- `df` — DataFrame containing the data
- `y_col` — Name of the target column
- `x_cols` — List of predictor columns
- `x_complexity_col` — Column for the x-axis in visualization plots
- `models_to_test` — Dictionary of `{name: {'model': estimator, 'params_grid': {...}}}`
- `select_features` — If True, applies Boruta feature selection before modeling

#### `@staticmethod select_features(df, y_col, x_cols)`

Preprocesses the dataset, creates dummy variables for categorical and binary columns, uses a Random Forest to evaluate feature importances, and applies the Boruta algorithm to select the most statistically significant features relative to the target.

The Boruta algorithm is a wrapper method that compares real feature importances against randomly generated "shadow" features across multiple iterations to determine which variables carry information useful for prediction.

**Returns:** Tuple of (selected_X DataFrame, y Series, list of dropped columns)

#### `estimate_optimization_time(self, time_per_combination)`

Estimates the total time required for model optimization by counting all hyperparameter combinations across all models and multiplying by the estimated time per combination.

#### `select_hyperparameters(self, regressor, params_grid)`

Performs GridSearchCV with the configured preprocessing pipeline and cross-validation strategy to find the best hyperparameters for a given regressor. Uses negative MSE as the scoring metric.

**Returns:** Tuple of (best fitted pipeline, best parameters dictionary)

#### `test_models_regression(self)`

Iterates through all models in `models_to_test`, optimizes hyperparameters for each, trains the best configuration on the training set, and evaluates on the test set. Stores results (model, params, predictions, true values) in `self.model_results`.

#### `grafico_fit_con_errore(self, model_name, h=16, w=9)`

Generates a 3-panel matplotlib figure for a specific trained model:
1. **Top panel:** Observed values (red dots) vs. predicted values (blue line) with standard error bands and 95% confidence intervals
2. **Middle panel:** Raw residuals as a bar chart
3. **Bottom panel:** Binned residuals showing mean residual per bin with 95% confidence intervals

---

### Boruta Feature Selection

### `class Boruta(BaseEstimator, TransformerMixin)`

**Module:** `scomp_link.models.regressor_optimizer`

Python implementation of the Boruta all-relevant feature selection algorithm. Unlike minimal optimal methods, Boruta finds ALL features carrying information useful for prediction, not just a compact subset.

#### `__init__(self, estimator, n_estimators=1000, perc=100, alpha=0.05, two_step=True, max_iter=100, random_state=None, verbose=1)`

**Parameters:**
- `estimator` — Estimator with `fit()` and `feature_importances_` (typically RandomForest)
- `n_estimators` — Number of trees, or `'auto'` for dynamic calculation
- `perc` — Percentile for shadow feature threshold comparison
- `alpha` — Significance level for p-value rejection
- `two_step` — Use FDR + Bonferroni two-step correction (more conservative)
- `max_iter` — Maximum iterations
- `random_state` — Random seed
- `verbose` — 0=silent, 1=iterations, 2=detailed

#### `fit(self, X, y)`

Run the Boruta selection algorithm. Iteratively creates shadow features (shuffled copies), trains the estimator on real + shadow features, and uses statistical tests to determine which real features are significantly more important than the best shadow features.

Sets `self.support_` (boolean mask of confirmed features), `self.support_weak_` (tentative features), and `self.ranking_` (feature rankings).

#### `transform(self, X, weak=False)`

Reduce input X to selected features. If `weak=True`, includes tentative features.

#### `fit_transform(self, X, y, weak=False)`

Fit Boruta and transform in one step.

#### `_fdrcorrection(self, pvals, alpha=0.05)` (private)

Benjamini/Hochberg FDR p-value correction. Included to avoid dependency on statsmodels.

---

### Classifier Optimizer

### `class ClassifierOptimizer`

**Module:** `scomp_link.models.classifier_optimizer`

Optimizes classification models via feature selection, hyperparameter tuning (GridSearchCV with StratifiedKFold), and evaluation of multiple classification algorithms.

#### `__init__(self, df, y_col, x_cols, models_to_test, select_features=False)`

Initialize the optimizer. Sets up preprocessing (OneHotEncoder + StandardScaler), stratified train/test split, and 5-fold stratified cross-validation. Optionally applies Boruta feature selection using a RandomForestClassifier.

#### `select_hyperparameters(self, classifier, params_grid)`

Performs GridSearchCV on the preprocessing pipeline + classifier, using accuracy scoring and stratified K-fold CV.

**Returns:** Tuple of (best fitted pipeline, best parameters dictionary)

#### `test_models_classification(self)`

Iterates through all models, optimizes hyperparameters, trains, and evaluates. Stores results including classification report, confusion matrix, and probability predictions (if available).

#### `print_results(self)`

Prints a formatted summary of all tested models including best parameters, classification report, and confusion matrix.

---

### Ensemble Optimizer

### `class EnsembleOptimizer`

**Module:** `scomp_link.models.ensemble_optimizer`

Combines multiple trained models using voting or stacking ensemble strategies.

#### `__init__(self, base_models, task_type='classification', strategy='voting')`

**Parameters:**
- `base_models` — List of `(name, model)` tuples with trained models
- `task_type` — `'classification'` or `'regression'`
- `strategy` — `'voting'` (average predictions) or `'stacking'` (meta-learner)

#### `create_ensemble(self, meta_model=None)`

Create the ensemble model. For voting: uses soft voting (classification) or average (regression). For stacking: uses 5-fold CV with an optional meta-learner.

#### `fit(self, X, y)`

Fit the ensemble model on training data. Creates the ensemble if not already created.

#### `predict(self, X)`

Make predictions with the fitted ensemble.

#### `evaluate_ensemble(self, X, y, cv=5)`

Evaluate ensemble performance using cross-validation. Returns dictionary with `mean_score`, `std_score`, and all `scores`.

---

### Anomaly Detector (Tabular)

### `class AnomalyDetector(BaseEstimator)`

**Module:** `scomp_link.models.anomaly_detector`

Multi-method anomaly detection for tabular data using consensus voting (majority rule) across 4 complementary approaches.

**Methods available:**
1. **Isolation Forest** — Tree-based, non-parametric isolation of outliers via random recursive partitioning
2. **Local Outlier Factor (LOF)** — Density-based, detects samples in sparse local neighborhoods
3. **TabNet Autoencoder** — Neural attention on tabular features; high reconstruction error = anomaly (requires `pytorch-tabnet`)
4. **Transformer Autoencoder** — Self-attention across features treated as tokens; high reconstruction error = anomaly (requires `torch`)

#### `__init__(self, contamination=0.05, methods=None, lof_neighbors=20, tabnet_epochs=50, transformer_epochs=80, transformer_d_model=32, transformer_nhead=4, transformer_num_layers=2, consensus_threshold=2, random_state=42, verbose=True)`

Initialize detector with contamination ratio (expected anomaly proportion) and method-specific parameters.

#### `fit_predict(self, df: pd.DataFrame, features: List[str]) -> Dict`

Run all configured methods on the specified numeric features and produce consensus labels. Scales data with StandardScaler, runs each method independently, then applies consensus voting.

**Returns:** Dictionary with:
- `'data'` — DataFrame with per-method anomaly columns, consensus score, and `is_anomaly` flag
- `'comparison'` — Summary table comparing each method's anomaly count
- `'consensus_threshold'` — Minimum methods required to flag an anomaly

#### `report(self, group_by: Optional[List[str]] = None) -> pd.DataFrame`

Generate a grouped summary of detected anomalies. If `group_by` is provided, aggregates anomaly counts and mean consensus scores by the specified columns.

---

### Time Series Anomaly Detector

### `class TimeSeriesAnomalyDetector`

**Module:** `scomp_link.models.ts_anomaly_detector`

Multi-method anomaly detection for univariate time series.

**Methods available:**
1. **Conv1D Autoencoder** — Learns normal temporal patterns from training data; flags high reconstruction error (requires `tensorflow`)
2. **Moving Average** — Flags deviations beyond N standard deviations from rolling mean
3. **Moving Median** — Robust variant using Median Absolute Deviation (MAD) instead of std
4. **ARIMA Residuals** — Flags large residuals from a fitted ARIMA model (requires `statsmodels`)

#### `__init__(self, methods=None, time_steps=288, ae_epochs=50, ae_batch_size=128, window_size=48, n_sigma=3.0, arima_order=(5,1,0), threshold_percentile=95.0, verbose=True)`

**Key Parameters:**
- `time_steps` — Sequence length for autoencoder (e.g., 288 = one day at 5-min intervals)
- `window_size` — Rolling window for moving average/median
- `n_sigma` — Number of standard deviations for threshold
- `threshold_percentile` — Percentile for autoencoder reconstruction error threshold

#### `fit(self, values: np.ndarray)`

Fit the detector on normal (non-anomalous) training data. Computes training mean/std and trains the autoencoder (if included in methods).

#### `detect(self, values: np.ndarray) -> Dict`

Detect anomalies in a time series.

**Returns:** Dictionary with:
- `'anomalies'` — Boolean array (True = anomaly, any method flagged it)
- `'methods'` — Dictionary of per-method boolean arrays
- `'consensus_score'` — Integer array (how many methods flagged each point)

---

### Contrastive Text Classifier

### `class ContrastiveTextClassifier`

**Module:** `scomp_link.models.contrastive_text`

Generalized contrastive learning text classifier using Siamese Networks with BERT. Suitable for text categorization with many classes, semantic similarity, few-shot learning, and scenarios with limited data per class.

#### `__init__(self, model_name='bert-base-uncased', use_faiss=True, embedding_dim=256)`

Initialize BERT tokenizer and model, create a ContrastiveSiameseModel, and optionally set up FAISS for fast nearest-neighbor inference.

#### `train_contrastive(self, df, text_col='text', label_col='label', epochs=5, batch_size=64, lr=2e-5, use_weighted_sampling=True, accumulation_steps=2, validation_split=0.1, early_stopping_patience=5)`

Train the model with contrastive learning. Creates positive/negative text pairs, uses differentiated learning rates (BERT layers get `lr`, projection layer gets `lr*10`), gradient accumulation, weighted sampling for class balance, and early stopping on validation loss. After training, computes label embeddings and builds a FAISS index.

#### `predict(self, text, top_k=1, return_confidence=False)`

Predict label(s) for a single text. Encodes the text through BERT + projection, then finds the nearest label embedding(s) via FAISS or cosine similarity.

**Returns:** Single label string (top_k=1) or list of labels, optionally with confidence scores.

#### `predict_batch(self, text_series, batch_size=512, top_k=1)`

Batch predict labels for multiple texts. Encodes all texts in batches, then performs bulk nearest-neighbor search.

**Returns:** DataFrame with columns: `text`, `prediction`, `confidence`, `top_k_predictions`, `top_k_confidences`.

#### `save(self, path='./ContrastiveTextModel')`

Save model weights (`model.pt`), metadata (`metadata.json` with labels, frequencies, and embedding_dim), and label embeddings (`embeddings.csv`).

#### `load(self, path='./ContrastiveTextModel', model_name='bert-base-uncased')`

Load a previously saved model. Reinitializes the architecture with the correct embedding dimension from metadata, loads weights, and rebuilds the FAISS index.

---

### Contrastive Network Components

**Module:** `scomp_link.models.contrastive_net`

#### `class ContrastiveSiameseModel(nn.Module)`

Siamese network with BERT backbone and a projection layer (Linear → ReLU → Dropout → Linear). Produces L2-normalized embeddings.

- `forward_one(input_ids, attention_mask)` — Encode a single input through BERT [CLS] token + projection layer
- `forward(text_ids, text_mask, label_ids, label_mask)` — Forward pass for both text and label inputs simultaneously

#### `class ContrastiveLoss(nn.Module)`

Contrastive loss function. For positive pairs (label=1): minimizes distance. For negative pairs (label=0): pushes embeddings apart beyond the margin.

- `forward(emb1, emb2, labels)` — Compute loss given two embedding batches and pair labels

#### `class SiameseDataset(Dataset)`

PyTorch Dataset for generating contrastive pairs. Generates 50% positive pairs (same label) and 50% negative pairs (different label). Includes text augmentation via random word dropout.

- `get_sample_weights(df)` — Static method: calculate inverse-frequency sample weights for balanced sampling

#### `class EarlyStopping`

Training callback that stops training if validation loss doesn't improve for `patience` epochs.

- `__call__(val_loss)` — Returns True if training should stop

---

### Supervised Text

### `class SpacyEmbeddingModel`

**Module:** `scomp_link.models.supervised_text`

Text classification using spaCy NLP pipelines combined with contrastive embeddings. Supports 60+ languages via spaCy models.

#### `__init__(self, lan='en', model_name='bert-base-uncased')`

Initialize spaCy model for the specified language and a BERT-based contrastive model for embeddings. Automatically downloads the spaCy model if not present.

#### `_select_language(self, lan, dict_size='sm')` (private)

Load a spaCy language model and its stop words. Tries `{lan}_core_web_{size}` first, then `{lan}_core_news_{size}`, downloading if necessary. Supports all spaCy-supported languages.

#### `training(self, to_tag, tagged, categorie)`

Training wrapper. Splits data into train/test (67/33), then calls the CNN text categorizer.

#### `cnn_embedding_textcategorizer(self, texts, y_lab, texts_test, y_lab_test, classes, width=16, embed_size=75, patience=20, epoch=100, learn_rate=0.1, dropout=0.2, batch_size=8, ...)`

Train a CNN-based text categorizer using spaCy's built-in `textcat` pipeline component. Performs text extraction (verbs, nouns, proper nouns), creates category labels, and trains with mini-batch SGD and early stopping. Reports precision, recall, F1, and accuracy on both train and test sets.

**Returns:** Tuple of (trained nlp pipeline, dictionary of classification reports and confusion matrices)

#### `evaluate_textcat(self, tokenizer, textcat, texts, cats)`

Evaluate a text categorization model. Computes weighted precision, recall, F1, and accuracy.

#### `report_progress(self, epoch, best, losses, scores)`

Print formatted training progress (loss, precision, recall, F1).

---

### Supervised Image

### `class CNNImg(BaseEstimator, ClassifierMixin)`

**Module:** `scomp_link.models.supervised_img`

Simplified CNN image classifier wrapper with sklearn-compatible interface. Currently a placeholder — for production use, implement with TensorFlow/PyTorch.

#### `__init__(self, input_shape=(224, 224, 3), n_classes=None, **kwargs)`

#### `fit(self, X, y)`

Fit the CNN model on image data (n_samples × height × width × channels).

#### `predict(self, X)`

Predict class labels for images.

#### `predict_proba(self, X)`

Predict class probabilities for images.

---

### Unsupervised Text

### `class TextEmbeddingClustering(BaseEstimator, ClusterMixin)`

**Module:** `scomp_link.models.unsupervised_text`

Text clustering using sentence-transformer embeddings + KMeans or hierarchical clustering.

#### `__init__(self, model_name='all-MiniLM-L6-v2', n_clusters=5, method='kmeans')`

#### `fit(self, X, y=None)`

Encode texts into embeddings using the sentence transformer, then fit the clustering algorithm.

#### `predict(self, X)`

Encode new texts and predict their cluster labels.

#### `fit_predict(self, X, y=None)`

Encode, fit, and predict cluster labels in one step.

---

### Unsupervised Image

### `class ClusterImg(BaseEstimator, ClusterMixin)`

**Module:** `scomp_link.models.unsupervised_img`

Simplified image clustering wrapper using KMeans on flattened image features. For production, use CNN feature extraction before clustering.

#### `__init__(self, n_clusters=8, **kwargs)`

#### `fit(self, X, y=None)`

Flatten images (if multi-dimensional) and fit KMeans clustering.

#### `predict(self, X)`

Flatten images and predict cluster labels using the fitted KMeans model.

#### `fit_predict(self, X, y=None)`

Fit and predict in one step.

---

## Validation

### Validator

### `class Validator`

**Module:** `scomp_link.validation.model_validator`

Handles the modeling and validation phases (M4, C1–C4, V1–V3) of the workflow.

#### `__init__(self, model: Any)`

Initialize with any fitted model.

#### `k_fold_cv(self, X, y, k: int = 5)`

**C2: K-Fold Cross Validation** — Run K-fold CV and return scores array. Prints mean and standard deviation.

#### `loocv(self, X, y)`

**C1: Leave-One-Out Cross Validation (LOOCV)** — Run LOOCV (can be slow for large datasets). Returns all scores.

#### `evaluate(self, y_true, y_pred, task_type: str = "regression", y_proba=None) -> Dict[str, float]`

**V3: Model Evaluation Metrics**

- **Regression metrics:** MSE, RMSE, MAE, R²
- **Classification metrics:** Accuracy, F1 (weighted), Precision (weighted), Recall (weighted)

#### `generate_validation_report(self, y_true, y_pred, task_type="regression", y_proba=None, report_name="Validation_Report.html")`

Generate a self-contained interactive HTML validation report using Plotly.

**Regression reports include:** Metrics summary table, Observed vs Predicted scatter plot, Residuals histogram.

**Classification reports include:** Metrics summary table, Confusion matrix heatmap, Probability distributions per class (if probabilities available).

---

### Advanced Cross-Validation

### `class AdvancedCV`

**Module:** `scomp_link.validation.advanced_cv`

Provides advanced cross-validation strategies beyond standard K-Fold.

#### `@staticmethod loocv(model, X, y, scoring=None)`

Leave-One-Out Cross Validation. Returns dictionary with `method`, `mean_score`, `std_score`, `n_splits`, and `scores`.

#### `@staticmethod bootstrap(model, X, y, n_iterations=1000, test_size=0.3, random_state=42)`

Bootstrap resampling validation with out-of-bag evaluation. Creates `n_iterations` bootstrap samples, trains a cloned model on each, and evaluates on the held-out samples.

**Returns:** Dictionary with `method`, `mean_score`, `std_score`, `n_iterations`, `scores`, and `confidence_interval_95` (2.5th and 97.5th percentiles).

#### `@staticmethod evaluate_all(model, X, y, include_loocv=True, include_bootstrap=True, bootstrap_iterations=1000, scoring=None)`

Run all advanced validation strategies. Skips LOOCV automatically if dataset has more than 1000 samples (too slow).

---

## Utilities

### HTML Report Builder

### `class ScompLinkHTMLReport`

**Module:** `scomp_link.utils.report_html`

Programmatic HTML report builder with embedded Plotly/Highcharts charts, DataFrames, and images. Uses a stateful builder pattern — methods append content to an internal HTML string.

#### `__init__(self, title, font_family='Baloo 2', url_background_header='...', description='Automatic Report', author='scomp-link toolkit', language='en', main_color='#6E37FA', light_color='#9682FF', dark_color='#4614B4')`

Initialize report with title, styling, and color theme.

#### `open_section(self, section_title: str)`

Open a new collapsible section in the report. Sections must be closed before opening another.

#### `close_section(self)`

Close the currently open collapsible section.

#### `add_title(self, title: str)`

Add an H2 title element to the report.

#### `add_text(self, text: str)`

Add a paragraph of text to the report.

#### `add_graph_to_report(self, fig, title: str)`

Add a Plotly figure to the report. The figure is serialized to JSON and embedded as inline JavaScript.

#### `add_matplotlib_graph_to_report(self, fig, title: str, dpi: int = 150, img_format: str = 'png')`

Add a matplotlib figure as a base64-encoded image (PNG or SVG).

#### `add_image_to_report(self, image_path: str, title: str)`

Add a local image file (any format) as a base64-encoded inline image.

#### `add_many_plots_with_selection_box_to_report(self, figures_dict: dict, title: str, **kwargs)`

Add multiple Plotly figures with dropdown selection boxes. The user can switch between plots using combo boxes.

#### `add_dataframe(self, df: pd.DataFrame, title: str, limit_max=2000)`

Add a DataFrame as an HTML table with downloadable CSV functionality. Limited to 2000 rows. Tables are styled with the report's color theme.

#### `single_plotly(self, fig, title: str, plotdivid=None) -> str`

Convert a Plotly figure to an HTML snippet with responsive JavaScript. Returns the raw HTML string (for internal use or custom embedding).

#### `select_plotly(self, figures_dict: dict, title: str, labels='Choose a label') -> str`

Generate HTML with dropdown selectors that show/hide multiple Plotly plots. Supports single or multi-level filtering.

#### `save_html(self, file_name='export.html')`

Save the complete report as a self-contained HTML file.

#### `save_pdf(self, file_name='export.pdf')`

Save the report as PDF by rendering HTML in a headless Chromium browser (via Playwright). Automatically installs Chromium on first use. Ensures all JavaScript charts are rendered, opens all collapsed sections, and arranges plots in a 2-column grid for print.

---

### Plotly Utilities

**Module:** `scomp_link.utils.plotly_utils`

Standalone functions for creating common Plotly visualizations with consistent styling.

#### `histogram(variable_float_for_distribution, name_of_the_column, h=600)`

Create a single histogram with standard deviation error bars and cumulative distribution function overlay.

**Parameters:**
- `variable_float_for_distribution` — Numeric array (cleaned of NaN/None)
- `name_of_the_column` — Label for the distribution
- `h` — Height of the plot in pixels

#### `multiple_histograms(variable_float_for_distribution, category_variable, category_name='x_label', y_label='y_label', h=300)`

Create stacked subplots of histograms, one per category. Each subplot includes: histogram, mean line, standard deviation error bars, and cumulative distribution function. Limited to 10 categories maximum.

**Parameters:**
- `variable_float_for_distribution` — Numeric array
- `category_variable` — Categorical array (same length) defining groups
- `category_name` — X-axis label
- `y_label` — Y-axis label
- `h` — Height per subplot in pixels

#### `barchart(categories, metric_values_list, x_axis_title='Category', y_axis_titles=None, order='asc', categorysorted=None, metric_values_line_list=None, y_line_axis_titles=None, percentage_y=True)`

Create a multi-panel bar chart with optional line overlays. Each element of `metric_values_list` creates a separate subplot. Supports custom sort order and optional line traces.

#### `linechart(date_list, lines, title_text='Trend analysis', x_label='date', y_labels='value', format_date="%Y-%m-%d", yaxis_ticksuffix='')`

Create a multi-series line chart with consistent styling. Parses dates from string format and applies the scomp-link color palette.

#### `area_chart(date_list, lines, title_text='Trend analysis', x_label='date', y_labels='value', format_date="%Y-%m-%d", yaxis_ticksuffix='')`

Create a stacked area chart. Same interface as `linechart` but with stacked filled areas.

---

### Highcharts Utilities

**Module:** `scomp_link.utils.highcharts`

Functions that generate raw HTML/JavaScript snippets for Highcharts-based charts, designed for embedding in HTML reports.

#### `streamgraphs(title, dates, series_dict, annotation=None, area=True)`

Generate a streamgraph (or stacked area chart) using Highcharts.

**Parameters:**
- `title` — Chart title
- `dates` — List of date strings for the x-axis
- `series_dict` — Dictionary of `{series_name: list_of_values}`
- `annotation` — Optional dictionary of `{annotation_text: date_index}` for chart annotations
- `area` — If True, renders as stacked area chart; if False, renders as streamgraph

**Returns:** HTML string with embedded Highcharts JavaScript.

#### `calendar_heatmap(title, series_dict, min=0, max=1)`

Generate a calendar heatmap (GitHub-style contribution grid) using Highcharts.

**Parameters:**
- `title` — Chart title
- `series_dict` — Dictionary of `{"yyyy-mm-dd": percentage_value}` (values 0–100)
- `min` — Minimum scale value
- `max` — Maximum scale value

**Returns:** HTML string with embedded Highcharts JavaScript.

#### `calendar_gantt(title, series_dict, min_date, max_date)`

Generate a Gantt chart using Highcharts Gantt module.

**Parameters:**
- `title` — Chart title
- `series_dict` — Dictionary defining tasks and their date ranges
- `min_date` — Earliest date on the chart
- `max_date` — Latest date on the chart

**Returns:** HTML string with embedded Highcharts Gantt JavaScript.

---

### PDF Converter

**Module:** `scomp_link.utils.pdf_converter`

Utility functions for converting Markdown and HTML files to PDF format.

#### `markdown_to_pdf(input_path: str, output_path: str = None, css: str = None) -> str`

Convert a Markdown file to PDF. Uses the `markdown` library for HTML conversion and `weasyprint` for PDF rendering.

**Parameters:**
- `input_path` — Path to the .md file
- `output_path` — Output PDF path (defaults to same name with .pdf extension)
- `css` — Optional custom CSS for styling

**Returns:** Path to the generated PDF file.

#### `html_to_pdf(input_path: str, output_path: str = None, css: str = None) -> str`

Convert an HTML file to PDF using WeasyPrint. Optionally injects additional CSS before rendering.

**Returns:** Path to the generated PDF file.

---

## Decorator

### `@print_throughput` (Module: `scomp_link.models.contrastive_text`)

Decorator that prints processing speed (items/second) for batch operations. Wraps functions that accept a `text_series` parameter, prints the count of items being processed, measures elapsed time, and reports throughput.
