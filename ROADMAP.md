# scomp-link Production Readiness Roadmap

Technical debt and architectural improvements to bring scomp-link to production grade.
Issues are grouped by priority and ordered by implementation dependency.

---

## Phase 1 — Foundation & Type Safety

### 1. Unified Exception Hierarchy

**File:** `scomp_link/exceptions.py` (new)

Define a custom exception tree so callers can catch specific failures:

```
ScompLinkError (base)
├── DataValidationError     — bad path, unsupported format, missing column
├── ModelTrainingError      — training failed, unsupported task type
├── ArtifactError           — save/load failed, version mismatch, pickle security
├── DriftDetectionError     — reference/current shape mismatch
└── UpdateError             — pip install failed, no network, permission denied
```

**What to change:**
- Replace all `sys.exit(...)` in `cli.py` with `raise DataValidationError(...)` caught at the `main()` entrypoint
- Replace silent `return None` / bare `except Exception: pass` in `mcp_server.py` with explicit exception raises
- Surface clean error messages in MCP tools via `{"status": "error", "error": str(e), "type": type(e).__name__}`

---

### 2. Input/Output Schemas (Pydantic)

**File:** `scomp_link/schemas.py` (new)

Pydantic v2 models for all CLI/MCP inputs and outputs:

```python
class TrainConfig(BaseModel):
    data: FilePath
    target: str
    task: Literal["regression", "classification", "clustering"]
    engineer: bool = False
    tune: bool = False
    n_trials: int = Field(50, ge=1, le=500)
    save_artifact: Path | None = None

class TrainResult(BaseModel):
    status: Literal["success", "error"]
    model_type: str
    metrics: dict[str, float]
    artifact_path: str | None = None
    error: str | None = None
```

**What to change:**
- Validate all inputs in `mcp_server.py` tool functions before executing core logic
- Return serialized `BaseModel` instances instead of hand-rolled `json.dumps({...})`
- Validate inputs in `cli.py` using the same schemas (single source of truth)

---

### 3. Shared Service Layer

**File:** `scomp_link/services.py` (new)

Extract duplicated business logic from `cli.py` and `mcp_server.py` into pure functions:

```python
def run_train(config: TrainConfig) -> TrainResult: ...
def run_predict(config: PredictConfig) -> PredictResult: ...
def run_validate(config: ValidateConfig) -> ValidateResult: ...
def run_drift(config: DriftConfig) -> DriftResult: ...
def run_describe(config: DescribeConfig) -> DescribeResult: ...
```

**What to change:**
- `cli.py` becomes a thin argparse wrapper that builds config objects and calls `services.py`
- `mcp_server.py` becomes a thin MCP wrapper that does the same
- All ML logic lives in `services.py`, tested once, used everywhere
- Eliminates ~400 LOC of duplication between the two entrypoints

---

### 4. Test Coverage > 80%

**Current state:** 25% minimum (fail_under = 25 in pyproject.toml)

**Files to add/expand:**

`tests/test_services.py` — unit tests for every function in `services.py`
`tests/test_integration.py` — end-to-end workflow:
  - describe → engineer → train → validate → predict → export
  - describe → train → detect_drift
  - describe → cluster_data
`tests/test_exceptions.py` — verify correct exceptions raised for bad inputs
`tests/test_artifacts.py` — save/load round-trip, version mismatch handling

**What to change in pyproject.toml:**
```toml
[tool.coverage.report]
fail_under = 80  # raise from 25
```

---

## Phase 2 — Architecture & Quality

### 5. Optional Dependency Groups

**Current problem:** `pip install scomp-link` pulls torch + tensorflow + transformers (~5 GB) even if the user only needs tabular ML.

**Refactor `pyproject.toml`:**

```toml
[project]
dependencies = [
    # Core only — pandas, scikit-learn, numpy, plotly, tqdm
    "pandas>=2.0.0",
    "scikit-learn>=1.3.0",
    "numpy>=1.26.0",
    "plotly>=5.0.0",
    "tqdm>=4.50.0",
    "optuna>=3.0.0",
    "polars>=0.20.0",
]

[project.optional-dependencies]
nlp     = ["torch", "transformers", "faiss-cpu", "sentence-transformers", "spacy"]
cv      = ["tensorflow", "pillow", "tf-keras"]
anomaly = ["pytorch-tabnet2", "statsmodels"]
reports = ["weasyprint", "playwright", "markdown", "PyJWT"]
shap    = ["shap", "lime"]
mcp     = ["mcp>=1.0.0,<2.0.0"]
serve   = ["flask>=2.0.0,<4.0.0"]
all     = ["scomp-link[nlp,cv,anomaly,reports,shap,mcp,serve]"]
dev     = ["pytest", "pytest-cov", "pre-commit"]
```

**What to change in code:**
- All imports of `torch`, `tensorflow`, `transformers` stay inside method bodies (already done via lazy imports — keep this)
- `_load_df`, `describe_data`, `train_model` (tabular), `detect_drift`, `cluster_data`, `forecast_series` must work with zero optional deps installed
- Add `check_deps` warnings when optional deps are missing for a specific tool

---

### 6. MCP Tool JSON Schemas & TypedDicts

Every MCP tool should declare its return type as a TypedDict and expose a JSON Schema.

**What to change in `mcp_server.py`:**
```python
class DescribeDataResult(TypedDict):
    shape: list[int]
    columns: list[ColumnStats]

class ColumnStats(TypedDict):
    column: str
    dtype: str
    missing_pct: float
    unique: int
    min: NotRequired[float]
    max: NotRequired[float]
    mean: NotRequired[float]
    std: NotRequired[float]
```

FastMCP supports passing `output_schema` to `@mcp.tool()` — use it.

---

### 7. Structured Logging

**Current problem:** Mix of `print()` statements and `logging.getLogger()` with inconsistent levels.

**What to change:**
- Replace `utils/logger.py` implementation with `structlog` or `loguru`
- Configure JSON output when `SCOMP_LOG_FORMAT=json` env var is set
- Add correlation ID to MCP request context (generate UUID per tool invocation)
- Remove all bare `print()` statements from core logic (keep them only in CLI for user-facing output)
- Add `SCOMP_LOG_LEVEL` env var support

```python
# Target pattern
logger.info("model.trained", model_type=model_type, duration_s=elapsed, n_rows=len(df))
# Not
print(f"Testing {model_name}:", datetime.now().strftime(...))
```

---

### 8. Jinja2 Templates for HTML Reports

**Current problem:** `ScompLinkHTMLReport` builds HTML via string concatenation — fragile, untestable, injection-prone.

**What to change:**
- Add `scomp_link/templates/` directory with `.html.j2` template files
- Replace string concatenation in `report_html.py` with `jinja2.Environment`
- Enables:
  - Unit testing individual template components
  - HTML validity checking
  - Clean separation of data from presentation
  - User-overridable templates

```python
# Target pattern
env = Environment(loader=PackageLoader("scomp_link", "templates"))
template = env.get_template("report_base.html.j2")
html = template.render(sections=self.sections, theme=self.theme)
```

---

## Phase 3 — Polish & Security

### 9. Artifact Security & Versioning

**Issue:** `.scomp` files use `pickle` internally. Loading untrusted artifacts = arbitrary code execution (RCE risk).

**What to add in `persistence/artifact.py`:**

```python
ARTIFACT_FORMAT_VERSION = 2

# On save — embed version header
metadata = {
    "format_version": ARTIFACT_FORMAT_VERSION,
    "scomp_link_version": __version__,
    "created_at": datetime.now().isoformat(),
    "python_version": sys.version,
    "task_type": config.get("task_type"),
}

# On load — check version and warn
if meta["format_version"] < ARTIFACT_FORMAT_VERSION:
    warnings.warn(
        f"Artifact was created with format v{meta['format_version']}, current is v{ARTIFACT_FORMAT_VERSION}. "
        "Some fields may be missing.",
        ArtifactVersionWarning,
        stacklevel=2,
    )
```

**Security warning to add:**
- Print/log a warning when loading any `.scomp` file: *"Only load artifacts from trusted sources — they use pickle serialization."*
- Document this clearly in README and docstrings
- Long term: add `--verify-signature` flag + optional HMAC signing

---

### 10. Thread Safety for MCP Global State

**Issue:** `_update_checked_this_session` is a module-level bool. Under concurrent FastMCP requests it's a race condition.

**What to change in `mcp_server.py`:**
```python
import threading
_session_lock = threading.Lock()
_update_checked_this_session = False

def _maybe_auto_update():
    global _update_checked_this_session
    with _session_lock:
        if _update_checked_this_session:
            return None
        _update_checked_this_session = True
    # ... rest of update logic
```

---

### 11. Auto-Update: Graceful Permission Errors

**Issue:** `_perform_update()` fails generically in read-only environments (Docker, HPC, venv without write access).

**What to change in `mcp_server.py`:**
```python
def _perform_update() -> dict:
    try:
        result = subprocess.run([sys.executable, "-m", "pip", "install", ...])
        ...
    except PermissionError:
        return {
            "success": False,
            "error": "permission_denied",
            "message": "Cannot upgrade in this environment (read-only site-packages). "
                       "Run manually: pip install --upgrade scomp-link[mcp]",
        }
    except subprocess.TimeoutExpired:
        return {"success": False, "error": "timeout", "message": "Update timed out after 120s"}
    except Exception as e:
        return {"success": False, "error": "unknown", "message": str(e)}
```

---

### 12. Strict Type Checking

**What to change:**
- Update `pyrightconfig.json` to enable strict mode:
```json
{
  "typeCheckingMode": "strict",
  "reportMissingTypeStubs": false,
  "reportUnknownVariableType": false
}
```
- Add `mypy` as alternative/complementary checker in CI
- Fix all resulting type errors (estimate: 200-300 fixes across the codebase)
- Add `pyright` step to `.github/workflows/ci.yml`

---

### 13. Configurable Timeouts & Memory Guards

**What to change in `mcp_server.py`:**

```python
import signal
from contextlib import contextmanager

MAX_DATASET_ROWS_WARNING = 1_000_000
MAX_DATASET_MB_WARNING = 500

def _load_df(path: str) -> pd.DataFrame:
    ...
    size_mb = Path(path).stat().st_size / 1_048_576
    if size_mb > MAX_DATASET_MB_WARNING:
        logger.warning("large_dataset", path=path, size_mb=round(size_mb, 1))
    return df
```

For long-running tools (`train_model`, `tune`), expose a `timeout_seconds` parameter that uses `signal.alarm` (Unix) or `concurrent.futures` with timeout (cross-platform).

---

## Implementation Order

| # | Item | Effort | Impact | Depends on |
|---|------|--------|--------|-----------|
| 1 | Exception hierarchy | S | High | — |
| 2 | Pydantic schemas | M | High | 1 |
| 3 | Service layer | L | High | 1, 2 |
| 4 | Test coverage | L | High | 3 |
| 5 | Optional deps | M | High | — |
| 6 | MCP JSON schemas | S | Medium | 2 |
| 7 | Structured logging | M | Medium | — |
| 8 | Jinja2 templates | L | Medium | — |
| 9 | Artifact security | S | High | — |
| 10 | Thread safety | S | Medium | — |
| 11 | Auto-update errors | S | Low | — |
| 12 | Strict type checking | L | Medium | 2, 3 |
| 13 | Timeouts/memory | M | Medium | — |

**Effort:** S = 1-2 days, M = 3-5 days, L = 1-2 weeks

---

## Non-Goals

These are explicitly out of scope for this roadmap:
- Changing the public API surface (breaking changes)
- Migrating from FastMCP to a custom MCP server
- Replacing scikit-learn with a different ML backend
- Adding new ML capabilities (this roadmap is about quality, not features)
