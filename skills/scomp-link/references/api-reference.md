# Python API Reference

## Core Pipeline

### `ScompLinkPipeline`

```python
from scomp_link import ScompLinkPipeline

pipe = ScompLinkPipeline("Project Name")
pipe.set_objectives(["Minimize RMSE"])
pipe.import_and_clean_data(df)                    # pandas DataFrame
pipe.select_variables(target_col='y', feature_cols=['x1', 'x2'])  # feature_cols optional
pipe.choose_model("numerical_prediction", metadata={})  # see model selection below
results = pipe.run_pipeline(
    task_type="regression",          # regression | classification | clustering | text | image | image_clustering
    test_size=0.2,
    models_to_test=None,             # dict for RegressorOptimizer/ClassifierOptimizer
    use_ensemble=False,              # voting/stacking ensemble
    ensemble_strategy='voting',      # voting | stacking
    advanced_cv=False,               # Enable LOOCV + Bootstrap
    cv_methods=['bootstrap'],        # loocv | bootstrap
    bootstrap_iterations=1000,
    # Text-specific:
    text_col=None,                   # column with text data
    use_contrastive=True,            # BERT contrastive vs TF-IDF
    text_model='bert-base-uncased',
    epochs=3, batch_size=32,
    # Image-specific:
    image_col=None,                  # column with image arrays
    # Clustering-specific:
    n_clusters=None,
)
pipe.save_model('./staging')
pipe.load_model('./staging')
predictions = pipe.predict(X_new)
```

**`choose_model` objective types:**
- `"numerical_prediction"` → auto-selects Ridge/Lasso/GBR based on data size
- `"categorical_known"` → auto-selects SVC/NaiveBayes/GBR based on features
- `"categorical_unknown"` → KMeans or MeanShift
- `"numerical_study"` → PCA / Geostatistical / UCM
- `"multi_numerical_prediction"` → VAR/VARMA or MLP

## Persistence

### `ScompArtifact`

```python
from scomp_link import ScompArtifact

# Save
artifact = ScompArtifact()
artifact.set_model(model)
artifact.set_preprocessor(scaler)                # optional sklearn transformer
artifact.set_config(task_type='regression', target_col='y', feature_cols=['x1', 'x2'])
artifact.set_metrics({'r2': 0.95, 'rmse': 2.1})
artifact.set_feature_schema(X_train)             # stores dtype/range per feature
artifact.set_sample_data(X_train, max_rows=200)  # for drift detection
artifact.set_metadata(author='team', version='1.0', description='...')
artifact.save('model.scomp')

# Load
loaded = ScompArtifact.load('model.scomp')
predictions = loaded.predict(X_new)              # chains preprocessor + model
info = loaded.info()                             # dict with all metadata
schema = loaded.feature_schema                   # dict {col: {dtype, min, max, mean}}
sample = loaded.sample_data                      # DataFrame

# Validate
ScompArtifact.is_scomp_file('model.scomp')      # returns bool
```

## Preprocessing

### `Preprocessor`

```python
from scomp_link import Preprocessor

prep = Preprocessor(df)                          # accepts pandas or polars
cleaned_df = prep.clean_data(remove_outliers=True, outlier_threshold=3.0)
integrated_df = prep.integrate_data(other_df, on='id', how='left')
features = prep.feature_selection(target_col='y', n_features=10)
eda = prep.run_eda()                             # returns dict with shape, missing, dtypes
X_train, X_test, y_train, y_test = prep.prepare_datasets(target_col='y', test_size=0.2)
```

### `FeatureEngineer`

```python
from scomp_link import FeatureEngineer

fe = FeatureEngineer(
    interactions=True,        # polynomial interactions between numeric features
    log_transform=True,       # log1p for skewed features (skew > threshold)
    skew_threshold=0.8,       # skewness threshold for log transform
    date_features=True,       # extract year/month/dow/weekend from date columns
    target_encode=True,       # target-encode high-cardinality categoricals
    target_encode_threshold=8,  # min unique values to trigger target encoding
    auto_bin=True,            # quantile binning
    n_bins=5,                 # number of bins
)
X_train_eng = fe.fit_transform(X_train, y_train)
X_test_eng = fe.transform(X_test)
```

### `DataQualityReport`

```python
from scomp_link import DataQualityReport

dqr = DataQualityReport(df)
report = dqr.generate()    # returns dict with: overview, missing, constants, duplicates, correlations, cardinality
dqr.save_html('quality.html')
```

## Models

### `RegressorOptimizer`

```python
from scomp_link import RegressorOptimizer
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import GradientBoostingRegressor

models_to_test = {
    'Ridge': {'model': Ridge(), 'params_grid': {'alpha': [0.1, 1.0, 10.0]}},
    'GBR': {'model': GradientBoostingRegressor(), 'params_grid': {'n_estimators': [50, 100], 'max_depth': [3, 5]}},
}

opt = RegressorOptimizer(df, 'target', x_cols, x_complexity_col='x1',
                         models_to_test=models_to_test, select_features=True)
opt.estimate_optimization_time(time_per_combination=2)
opt.test_models_regression()
opt.grafico_fit_con_errore('Ridge')  # matplotlib plot
# Results: opt.model_results dict with Model, Params, fitted values
```

### `ClassifierOptimizer`

```python
from scomp_link import ClassifierOptimizer

opt = ClassifierOptimizer(df, 'target', x_cols, models_to_test=models, select_features=True)
opt.test_models_classification()
opt.print_results()
# Results: opt.model_results dict with Model, Params, Report, Confusion_Matrix
```

### `AnomalyDetector`

```python
from scomp_link import AnomalyDetector

detector = AnomalyDetector(
    contamination=0.05,
    methods=['iforest', 'lof', 'tabnet', 'transformer'],
    consensus_threshold=2,
    tabnet_epochs=50,
    transformer_epochs=80,
)
results = detector.fit_predict(df, features=['col1', 'col2'])
# results['data'] — DataFrame with is_anomaly column
# results['comparison'] — method-by-method stats
```

### `TimeSeriesForecaster`

```python
from scomp_link import TimeSeriesForecaster

fc = TimeSeriesForecaster(method='auto', horizon=30, seasonal_period=12)
fc.fit(series)                          # pandas Series
forecast = fc.predict()                 # Series of predicted values
ci = fc.predict_with_ci(steps=10)       # DataFrame with forecast, lower, upper
cv = fc.walk_forward_cv(series, n_splits=5, horizon=12)  # dict with mean_mae, mean_rmse, mean_mape
```

### Tuning

```python
from scomp_link import OptunaOptimizer, HalvingSearchOptimizer, EarlyStoppingCV

# Optuna
def param_space(trial):
    return {'n_estimators': trial.suggest_int('n_estimators', 50, 500), ...}
opt = OptunaOptimizer(ModelClass, param_space, scoring='r2', n_trials=100)
best_model = opt.optimize(X_train, y_train)

# Halving
halving = HalvingSearchOptimizer(model, param_grid, scoring='r2')
best_model = halving.optimize(X_train, y_train)
```

## Validation

### `Validator`

```python
from scomp_link import Validator

val = Validator(model)
metrics = val.evaluate(y_true, y_pred, task_type='regression')  # mse, rmse, mae, r2
scores = val.k_fold_cv(X, y, k=5)
loocv_scores = val.loocv(X, y)
val.generate_validation_report(y_true, y_pred, task_type='regression', report_name='report.html')
```

### `FairnessMetrics`

```python
from scomp_link import FairnessMetrics

fm = FairnessMetrics(y_true, y_pred, sensitive_feature=gender_array)
report = fm.compute_all()      # demographic_parity, disparate_impact, equalized_odds
summary = fm.summary(report)   # DataFrame
fig = fm.plot_fairness_report(report)
```

## Monitoring

### `DriftDetector`

```python
from scomp_link import DriftDetector

detector = DriftDetector(reference_df, psi_threshold=0.2, ks_alpha=0.05)
report = detector.detect(current_df, features=['col1', 'col2'])  # optional feature subset
summary = detector.summary(report)   # dict: drifted_features, total_features, max_psi, worst_feature
fig = detector.plot_drift_report(report)
fig_dist = detector.plot_feature_distribution('col1', current_df)
```

## Explainability

```python
from scomp_link import ShapExplainer, LimeExplainer

# SHAP
shap_exp = ShapExplainer(model, X_background[:100])
shap_exp.explain(X_test)
importance = shap_exp.feature_importance()  # DataFrame: feature, importance
fig = shap_exp.plot_importance(top_n=10)

# LIME
lime_exp = LimeExplainer(model, X_train, task='regression')  # or 'classification'
exp = lime_exp.explain_instance(X_test.iloc[0], num_features=5)
exp.as_list()   # [(feature_rule, weight), ...]
fig = lime_exp.plot_explanation(exp)
```
