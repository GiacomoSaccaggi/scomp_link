# Persistence

Custom `.scomp` format for saving and loading complete ML pipelines.

## Modules

| Module | Class | Description |
|--------|-------|-------------|
| `artifact.py` | `ScompArtifact` | ZIP-based format: model + preprocessor + config + metrics + feature schema + sample data |

## .scomp File Structure

```
artifact.scomp (ZIP)
├── __magic__           # File identification bytes
├── manifest.json       # Version, timestamp, python/package versions
├── model.pkl           # Fitted model
├── preprocessor.pkl    # Fitted preprocessor (optional)
├── config.json         # Pipeline configuration
├── metrics.json        # Training/validation metrics
├── feature_schema.json # Column names, types, ranges
└── sample_data.parquet # Small training sample (for drift detection)
```

## Usage

```python
from scomp_link import ScompArtifact

# Save
artifact = ScompArtifact()
artifact.set_model(model).set_preprocessor(scaler)
artifact.set_config(task_type='regression', target_col='y')
artifact.set_metrics({'r2': 0.95})
artifact.set_feature_schema(X_train)
artifact.save('pipeline.scomp')

# Load and predict
loaded = ScompArtifact.load('pipeline.scomp')
predictions = loaded.predict(new_data)
```
