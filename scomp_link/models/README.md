# Models

Model selection, training, optimization, and specialized ML models.

## Modules

| Module | Class | Description |
|--------|-------|-------------|
| `model_factory.py` | `ModelFactory` | Decision-tree-based model selection |
| `regressor_optimizer.py` | `RegressorOptimizer` | Grid search + Boruta feature selection for regression |
| `classifier_optimizer.py` | `ClassifierOptimizer` | Grid search for classification |
| `ensemble_optimizer.py` | `EnsembleOptimizer` | Voting and stacking ensemble strategies |
| `advanced_tuning.py` | `OptunaOptimizer`, `HalvingSearchOptimizer`, `EarlyStoppingCV` | Bayesian optimization, successive halving, patience-based stopping |
| `forecaster.py` | `TimeSeriesForecaster` | ARIMA, SARIMA, Exponential Smoothing, walk-forward CV |
| `anomaly_detector.py` | `AnomalyDetector` | Multi-method consensus (IForest, LOF, TabNet, Transformer) |
| `ts_anomaly_detector.py` | `TimeSeriesAnomalyDetector` | Conv1D autoencoder, moving avg/median, ARIMA residuals |
| `contrastive_text.py` | `ContrastiveTextClassifier` | BERT contrastive learning + FAISS inference |
| `supervised_text.py` | `SpacyEmbeddingModel` | spaCy/sentence-transformers + KNN |
| `supervised_img.py` | `CNNImg` | TensorFlow/Keras CNN image classification |
| `unsupervised_img.py` | `ClusterImg` | Image feature extraction + clustering |

## Usage

```python
from scomp_link.models.advanced_tuning import OptunaOptimizer
from scomp_link import TimeSeriesForecaster, AnomalyDetector

# Optuna HPO
optimizer = OptunaOptimizer(GBR, param_space, scoring='r2', n_trials=100)
best = optimizer.optimize(X, y)

# Forecasting
fc = TimeSeriesForecaster(method='auto', horizon=30)
fc.fit(series)
forecast = fc.predict_with_ci()

# Anomaly detection
detector = AnomalyDetector(methods=['iforest', 'lof'], contamination=0.05)
results = detector.fit_predict(df, features=['x1', 'x2'])
```
