# Pipeline DSL (`>>` operator)

Build ML pipelines and HTML reports declaratively using the `>>` operator — Airflow-style composition with lazy execution.

---

## How It Works

```python
# >> builds the chain (lazy) — .run() executes it
result = (StepA >> StepB >> StepC).run()
```

- `>>` connects steps into a chain — nothing executes yet
- `.run()` triggers sequential execution
- Each step receives the output of the previous step
- Two chain types: **ML chains** and **Report chains** (cannot mix)

---

## ML Chains

Train ML models from data to results in one expression.

```python
from scomp_link import CleanStep, SelectStep, ModelStep, TrainStep
import pandas as pd

df = pd.read_csv("train.csv")

results = (
    CleanStep(df)
    >> SelectStep("price", features=["sqm", "rooms", "age"])
    >> ModelStep("numerical_prediction")
    >> TrainStep("regression", test_size=0.2)
).run()

print(results["metrics"])  # {'r2': 0.84, 'rmse': 24156.4, ...}
```

### ML Steps Reference

| Step | Parameters | What it does |
|------|-----------|-------------|
| `CleanStep(df)` | `df` — pandas DataFrame | Cleans data (duplicates, outliers, types) |
| `SelectStep(target, features=None)` | Target column, optional feature list | Selects variables for modeling |
| `ModelStep(objective)` | `"numerical_prediction"`, `"categorical_known"`, `"categorical_unknown"` | Auto-selects model based on data |
| `TrainStep(task, test_size=0.2)` | `"regression"`, `"classification"`, `"clustering"` | Trains, validates, returns metrics |

### Example: Classification

```python
from scomp_link import CleanStep, SelectStep, ModelStep, TrainStep

results = (
    CleanStep(df)
    >> SelectStep("churn", features=["tenure", "monthly_charges", "contract"])
    >> ModelStep("categorical_known")
    >> TrainStep("classification")
).run()

print(results["metrics"]["f1"])  # 0.87
```

---

## Report Chains

Build HTML reports step by step — same `>>` syntax.

```python
from scomp_link import SectionStep, TitleStep, TextStep, TableStep, GraphStep, SaveStep
from scomp_link.utils.report_html import ScompLinkHTMLReport

report = ScompLinkHTMLReport("Q4 Report")

(
    SectionStep("Executive Summary")
    >> TitleStep("Revenue Analysis")
    >> TextStep("Q4 showed 15% growth in all segments.")
    >> TableStep(revenue_df, "Revenue by Region")
    >> GraphStep(fig, "Trend Chart")
    >> SectionStep("Details")
    >> TableStep(details_df, "Full Breakdown")
    >> SaveStep("q4_report.html")
).run(report)
```

### Report Steps Reference

| Step | Parameters | What it does |
|------|-----------|-------------|
| `SectionStep(title)` | Section title | Opens a collapsible section (auto-closes previous) |
| `TitleStep(title)` | Heading text | Adds `<h2>` heading |
| `SubtitleStep(subtitle)` | Subheading text | Adds `<h3>` subheading |
| `TextStep(text)` | Paragraph text | Adds `<p>` paragraph |
| `TableStep(df, title)` | DataFrame + title | Adds interactive table (sortable, CSV export) |
| `GraphStep(fig, title)` | Plotly figure + title | Adds interactive Plotly chart |
| `RawGraphStep(svg, title)` | SVG string + title | Adds RAWGraphs SVG chart |
| `HighchartsStep(html, title)` | HTML snippet + title | Adds Highcharts chart |
| `CodeStep(...)` | Code + options | Adds syntax-highlighted code block |
| `DiffStep(...)` | Old/new code + options | Adds side-by-side diff view |
| `SaveStep(path)` | Output file path | Saves report to HTML file |
| `LogStep(label)` | Debug label | Prints intermediate state (no-op in output) |

---

## Code Blocks (`CodeStep`)

Add syntax-highlighted code to reports with copy-to-clipboard button.

```python
from scomp_link import CodeStep, SectionStep, SaveStep
from scomp_link.utils.report_html import ScompLinkHTMLReport

report = ScompLinkHTMLReport("Code Examples")

(
    SectionStep("Implementation")
    >> CodeStep("print('hello')", "python", "Hello World", output="hello")
    >> CodeStep("fn main() { println!(\"hi\"); }", "rust", "Rust Example")
    >> CodeStep(long_code, "python", "Full Module", line_numbers=True, collapsed=True)
    >> SaveStep("code_report.html")
).run(report)
```

### `CodeStep` Parameters

```python
CodeStep(
    code: str,              # Source code to display
    language: str = "python",  # Highlighting language (python, rust, js, sql, bash, json, yaml, etc.)
    title: str = "",        # Title above the code block
    output: str | None = None,  # Optional program output (terminal-style box below)
    line_numbers: bool = False,  # Show line numbers
    collapsed: bool = False,     # Wrap in collapsible <details> element
)
```

### Features

- **Syntax highlighting** — Prism.js with Tomorrow Night theme, 290+ languages via autoloader
- **Copy-to-clipboard** — button appears automatically on every code block
- **Line numbers** — toggle with `line_numbers=True`
- **Collapsible** — hides long code behind a toggle when `collapsed=True`
- **Terminal output** — dark box with monospace font for program output

### Supported Languages

Any language Prism.js supports (loaded on demand):

`python`, `rust`, `javascript`, `typescript`, `java`, `c`, `cpp`, `csharp`, `go`, `ruby`, `php`, `swift`, `kotlin`, `scala`, `r`, `sql`, `bash`, `powershell`, `docker`, `yaml`, `json`, `toml`, `xml`, `html`, `css`, `markdown`, `latex`, `matlab`, `julia`, `haskell`, `elixir`, `dart`, `lua`, `perl`, `zig`, ...

---

## Diff Views (`DiffStep`)

Show code changes side-by-side with GitHub-style coloring.

```python
from scomp_link import DiffStep, SectionStep, SaveStep
from scomp_link.utils.report_html import ScompLinkHTMLReport

report = ScompLinkHTMLReport("Code Review")

(
    SectionStep("Model Changes")
    >> DiffStep(
        "model = Ridge(alpha=1.0)",
        "model = GradientBoostingRegressor(n_estimators=200, max_depth=5)",
        "python",
        "Model Upgrade",
        "v1.py",
        "v2.py",
    )
    >> DiffStep(
        "test_size: 0.2\nmetric: mse",
        "test_size: 0.3\nmetric: rmse\ntuning: optuna",
        "yaml",
        "Config Change (collapsed)",
        collapsed=True,
    )
    >> SaveStep("review.html")
).run(report)
```

### `DiffStep` Parameters

```python
DiffStep(
    old_code: str,          # Original code (left side, deletions in red)
    new_code: str,          # Modified code (right side, additions in green)
    language: str = "python",  # Language for syntax highlighting
    title: str = "",        # Title above the diff
    old_label: str = "before",  # Label for old version
    new_label: str = "after",   # Label for new version
    collapsed: bool = False,    # Wrap in collapsible element
)
```

### Features

- **Side-by-side view** — old code left, new code right
- **Color-coded** — red background for deletions, green for additions
- **Syntax highlighting** — via highlight.js inside diff2html
- **Line matching** — intelligently matches similar lines across versions
- **Custom labels** — name your versions (e.g., "v1.0", "v2.0", "production", "candidate")
- **Collapsible** — hide large diffs behind a toggle
- **Handles identical code** — shows "No differences found" when old == new

---

## Mixing Sections, Code, and Diff

A complete code review report:

```python
from scomp_link import (
    SectionStep, TitleStep, TextStep, CodeStep, DiffStep, TableStep, SaveStep
)
from scomp_link.utils.report_html import ScompLinkHTMLReport
import pandas as pd

report = ScompLinkHTMLReport("Sprint 42 — Code Review")

metrics = pd.DataFrame([
    {"metric": "R²", "before": 0.78, "after": 0.84},
    {"metric": "RMSE", "before": 28400, "after": 24156},
])

(
    SectionStep("Summary")
    >> TextStep("Upgraded model from Ridge to GradientBoosting with Optuna tuning.")
    >> TableStep(metrics, "Performance Comparison")

    >> SectionStep("Code Changes")
    >> DiffStep(
        "model = Ridge(alpha=1.0)\nmodel.fit(X_train, y_train)",
        "study = optuna.create_study()\nstudy.optimize(objective, n_trials=50)\nmodel = GBR(**study.best_params)\nmodel.fit(X_train, y_train)",
        "python", "Training Pipeline", "sprint_41", "sprint_42"
    )

    >> SectionStep("Validation")
    >> CodeStep(
        "scomp-link validate --artifact model.scomp --data test.csv --target price",
        "bash", "Validation Command",
        output="R²: 0.8392 | RMSE: 24156.43 | ✅ Within thresholds"
    )

    >> SaveStep("sprint42_review.html")
).run(report)
```

---

## `LogStep` — Debugging

Insert `LogStep` anywhere to inspect the intermediate state without affecting the pipeline:

```python
from scomp_link import CleanStep, SelectStep, ModelStep, TrainStep, LogStep

results = (
    CleanStep(df)
    >> LogStep("after clean")      # prints pipeline state
    >> SelectStep("price")
    >> LogStep("after select")
    >> ModelStep("numerical_prediction")
    >> TrainStep("regression")
).run()
```

`LogStep` works in both ML and Report chains — it's neutral and doesn't change the chain type.

---

## Rules & Constraints

1. **Cannot mix ML and Report steps** in the same chain — raises `TypeError` immediately
2. **ML chains**: call `.run()` with no arguments — returns a results dict
3. **Report chains**: call `.run(report)` passing a `ScompLinkHTMLReport` instance
4. **`SaveStep`** auto-closes any open section before saving
5. **`SectionStep`** auto-closes the previous section (if using the Python API directly, you must call `close_section()` manually — the DSL handles it for you)
6. **`LogStep`** is the only step that works in both chain types

---

## Comparison: 3 Ways to Build Reports

The same report can be built with any of the 3 interfaces:

### 1. Python API (direct method calls)

```python
report = ScompLinkHTMLReport("Report")
report.open_section("Code")
report.add_code_block("x = 1", "python", "Example", output="1")
report.add_diff("a = 1", "a = 2", "python", "Change")
report.close_section()
report.save_html("out.html")
```

### 2. Pipeline DSL (`>>` operator)

```python
report = ScompLinkHTMLReport("Report")
(
    SectionStep("Code")
    >> CodeStep("x = 1", "python", "Example", output="1")
    >> DiffStep("a = 1", "a = 2", "python", "Change")
    >> SaveStep("out.html")
).run(report)
```

### 3. MCP Tools (for AI agents)

```
report_create("Report") → report_id
report_add_section(report_id, "Code")
report_add_code(report_id, "x = 1", "python", "Example", "1")
report_add_diff(report_id, "a = 1", "a = 2", "python", "Change")
report_save(report_id, "out.html")
```

All three produce identical HTML output.
