# Quick Start

## Installation

```bash
pip install scomp-link
```

Requires Python 3.10+. Optional extras:
```bash
pip install scomp-link[mcp]     # MCP server for AI agents
pip install scomp-link[serve]   # Flask for REST API serving
```

---

## 5-Minute CLI Tutorial

```bash
# 1. Scaffold a project
scomp-link init my_project && cd my_project

# 2. Profile your data (put a CSV in data/)
scomp-link describe --data data/dataset.csv

# 3. Train a model
scomp-link run --data data/dataset.csv --target price --task regression --save-artifact models/model.scomp

# 4. Validate
scomp-link validate --artifact models/model.scomp --data data/test.csv --target price --format table

# 5. Predict
scomp-link predict --artifact models/model.scomp --data data/new_data.csv --output predictions.csv

# 6. Deploy
scomp-link serve --artifact models/model.scomp --port 8080
```

---

## 5-Minute Python Tutorial

```python
from scomp_link import ScompLinkPipeline, ScompArtifact, set_verbosity
import pandas as pd

set_verbosity("info")

# Load data
df = pd.read_csv("train.csv")

# Build pipeline
pipe = ScompLinkPipeline("My Project")
pipe.set_objectives(["Minimize RMSE"])
pipe.import_and_clean_data(df)
pipe.select_variables(target_col='price')
pipe.choose_model("numerical_prediction")
results = pipe.run_pipeline(task_type="regression")

print(results['metrics'])  # {'mse': ..., 'rmse': ..., 'mae': ..., 'r2': ...}

# Save
artifact = ScompArtifact()
artifact.set_model(pipe.model)
artifact.set_config(task_type='regression', target_col='price')
artifact.set_metrics(results['metrics'])
artifact.save('model.scomp')

# Load and predict
loaded = ScompArtifact.load('model.scomp')
predictions = loaded.predict(new_data)
```

---

## YAML Pipeline (No Code)

```yaml
# pipeline.yaml
data: train.csv
target: price
task: regression
engineer:
  interactions: true
  log_transform: true
tune:
  method: optuna
  n_trials: 50
validate:
  test_data: test.csv
  report: validation.html
save: model.scomp
```

```bash
scomp-link pipeline --config pipeline.yaml
```

---

## AI Agent Integration

```bash
# Option 1: MCP Server (structured tools)
pip install scomp-link[mcp]
scomp-link mcp  # starts stdio server

# Option 2: Agent Skill (documentation-based)
cp -r skills/scomp-link ~/.kiro/skills/
```

See [Agent Integration Guide](agent-integration.md) for full setup.

---

## Next Steps

- [CLI Reference](cli.md) — All 25 commands
- [Python API](API_REFERENCE.md) — Programmatic usage
- [Visualization Guide](visualization.md) — 39 chart types
- [Examples](examples.md) — 34 runnable examples
