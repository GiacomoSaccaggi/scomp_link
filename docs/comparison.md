# scomp-link vs Alternatives

## Feature Comparison

| Feature | scomp-link | sklearn (raw) | PyCaret | AutoGluon |
|---------|-----------|---------------|---------|-----------|
| **Zero-code CLI** | ✅ 25 commands | ❌ Python only | ⚠️ Limited | ❌ Python only |
| **Lines to train + validate** | 2 (`run` + `validate`) | ~30 | ~10 | ~8 |
| **Auto model selection** | ✅ Decision tree | ❌ Manual | ✅ | ✅ |
| **Hyperparameter tuning** | ✅ Optuna + Halving | ❌ Manual | ✅ | ✅ |
| **Model persistence** | ✅ `.scomp` (model+config+metrics) | ⚠️ pickle only | ✅ | ✅ |
| **HTML reports** | ✅ 39 chart types | ❌ | ✅ Basic | ❌ |
| **Data drift detection** | ✅ PSI + KS | ❌ | ❌ | ❌ |
| **Anomaly detection** | ✅ 4 methods (IForest+LOF+TabNet+Transformer) | ❌ | ❌ | ❌ |
| **Fairness metrics** | ✅ Demographic parity, DI, equalized odds | ❌ | ❌ | ❌ |
| **Time series** | ✅ ARIMA + ETS + auto | ❌ | ✅ | ✅ |
| **Text classification** | ✅ TF-IDF + BERT contrastive | ❌ | ✅ Basic | ✅ |
| **REST API deploy** | ✅ `scomp-link serve` | ❌ | ❌ | ❌ |
| **MCP server (AI agents)** | ✅ 15 tools | ❌ | ❌ | ❌ |
| **YAML pipelines** | ✅ `scomp-link pipeline` | ❌ | ❌ | ❌ |
| **Agent Skill** | ✅ SKILL.md | ❌ | ❌ | ❌ |
| **Feature engineering** | ✅ Auto (interactions, log, encoding) | ❌ Manual | ✅ | ✅ |
| **SHAP + LIME** | ✅ Built-in | ❌ Separate install | ✅ SHAP only | ✅ SHAP only |
| **Production monitoring** | ✅ `scomp-link monitor` | ❌ | ❌ | ❌ |
| **Install size** | ~200MB (all deps) | ~30MB | ~500MB | ~2GB |
| **Import time** | ~6ms (lazy loading) | ~1s | ~5s | ~10s |
| **Python version** | 3.10+ | 3.9+ | 3.8+ | 3.8+ |

## Code Comparison: Train + Validate + Deploy

### scomp-link (CLI — 3 lines)

```bash
scomp-link tune --data train.csv --target price --task regression --method optuna --n-trials 50 --save-artifact model.scomp
scomp-link validate --artifact model.scomp --data test.csv --target price --report report.html
scomp-link serve --artifact model.scomp --port 8080
```

### scomp-link (Python — 8 lines)

```python
from scomp_link import ScompLinkPipeline, ScompArtifact

pipe = ScompLinkPipeline("Prices")
pipe.import_and_clean_data(df)
pipe.select_variables(target_col='price')
pipe.choose_model("numerical_prediction")
results = pipe.run_pipeline(task_type="regression", advanced_cv=True)
ScompArtifact().set_model(pipe.model).set_metrics(results['metrics']).save('model.scomp')
```

### sklearn (Python — 35 lines)

```python
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
import pickle

df = pd.read_csv('train.csv')
X = df.drop('price', axis=1)
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

pipe = Pipeline([('scaler', StandardScaler()), ('model', GradientBoostingRegressor())])
params = {'model__n_estimators': [50, 100, 200], 'model__max_depth': [3, 5, 8]}
grid = GridSearchCV(pipe, params, cv=5, scoring='r2')
grid.fit(X_train, y_train)

y_pred = grid.predict(X_test)
print(f"R2: {r2_score(y_test, y_pred):.4f}")
print(f"RMSE: {mean_squared_error(y_test, y_pred)**0.5:.4f}")

with open('model.pkl', 'wb') as f:
    pickle.dump(grid.best_estimator_, f)
# No built-in serve, no HTML report, no drift detection...
```

### PyCaret (Python — 6 lines)

```python
from pycaret.regression import setup, compare_models, predict_model, save_model

s = setup(data=df, target='price', session_id=42)
best = compare_models()
predictions = predict_model(best, data=test_df)
save_model(best, 'model')
# No CLI, no serve, no drift, no fairness, no MCP...
```

## When to Use What

| Use Case | Best Tool |
|----------|-----------|
| Quick prototype from terminal | **scomp-link** |
| Production ML with monitoring | **scomp-link** |
| AI agent automation | **scomp-link** (only one with MCP) |
| Interactive dashboards + reports | **scomp-link** (39 chart types) |
| Deep custom model architectures | sklearn / PyTorch directly |
| Kaggle competitions | AutoGluon |
| Enterprise AutoML (no code) | PyCaret |
| Ultra-large datasets (100M+ rows) | Spark / Dask + sklearn |
