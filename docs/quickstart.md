# Quick Start

## Installation

```bash
pip install scomp-link
```

## Regression Pipeline

```python
from scomp_link import ScompLinkPipeline
import pandas as pd
import numpy as np

# Create data
N = 1000
df = pd.DataFrame({
    'x1': np.random.randn(N),
    'x2': np.random.randn(N),
    'y':  2*np.random.randn(N) + 0.5
})

# Build and run
pipe = ScompLinkPipeline("Demo")
pipe.set_objectives(["Minimize RMSE"])
pipe.import_and_clean_data(df)
pipe.select_variables(target_col='y')
pipe.choose_model("numerical_prediction",
                  metadata={"only_numerical_exogenous": True,
                            "all_variables_important": False})
results = pipe.run_pipeline(task_type="regression")
```

## Silence Output

```python
import scomp_link
scomp_link.set_verbosity("silent")
```

## Save a Pipeline

```python
from scomp_link import ScompArtifact

artifact = ScompArtifact()
artifact.set_model(model).set_preprocessor(scaler)
artifact.set_config(task_type='regression', target_col='y')
artifact.set_metrics({'r2': 0.95})
artifact.save('my_pipeline.scomp')

# Load later
loaded = ScompArtifact.load('my_pipeline.scomp')
predictions = loaded.predict(new_data)
```
