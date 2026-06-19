# Contributing to scomp-link

*Welcome to the Rebel Alliance, pilot.* 🚀

We're glad you want to contribute. Whether you're fixing a bug, adding a feature, or improving docs — every contribution strengthens the Force.

## The Jedi Code (Ground Rules)

1. **Be respectful.** We follow the [Code of Conduct](CODE_OF_CONDUCT.md).
2. **One commit, one purpose.** Keep PRs focused. A bugfix is not a refactor.
3. **Tests or it didn't happen.** New features need tests. Bug fixes need a test that reproduces the bug.
4. **English only** in code, comments, docstrings, and commit messages.

## Joining a Mission (How to Contribute)

### 1. Report a Bug 🐛

Use the [Bug Report template](https://github.com/GiacomoSaccaggi/scomp_link/issues/new?template=bug_report.md). Include:
- What you expected vs what happened
- Minimal reproduction steps
- Python version and OS

### 2. Request a Feature 💡

Use the [Feature Request template](https://github.com/GiacomoSaccaggi/scomp_link/issues/new?template=feature_request.md). Explain the use case, not just the solution.

### 3. Submit a Pull Request 🛠️

```bash
# Clone and install
git clone https://github.com/GiacomoSaccaggi/scomp_link.git
cd scomp_link
pip install -e ".[dev]"
pre-commit install

# Create a branch
git checkout -b feat/my-awesome-feature

# Make changes, add tests, run them
pytest tests/ -v

# Commit (conventional commits preferred)
git commit -m "feat: add lightsaber mode to FeatureEngineer"

# Push and open PR
git push origin feat/my-awesome-feature
```

### Commit Message Format

```
<type>: <short description>

Types: feat, fix, docs, test, refactor, chore, ci
```

## Coding Standards

- **Style**: ruff (configured in `.pre-commit-config.yaml`)
- **Docstrings**: English, with `Dependencies:`, `PARAMETERS:`, `Usage example:` sections
- **Logging**: Use `from scomp_link.utils.logger import get_logger` — never `print()`
- **Imports**: Heavy optional deps (torch, tensorflow) imported inside method bodies, never top-level
- **Tests**: pytest, synthetic data only (no external datasets), `random_state=42`

## Architecture (Know Your Ship)

```
scomp_link/
├── cli.py              # 13 CLI commands
├── core.py             # Pipeline orchestrator
├── preprocessing/      # Preprocessor, FeatureEngineer, DataQualityReport
├── models/             # All model classes + tuning
├── validation/         # Metrics, CV, fairness
├── explainability/     # SHAP, LIME
├── monitoring/         # Drift detection
├── persistence/        # .scomp format
└── utils/              # Logger, decorators, reports, charts
```

## Running the Test Suite

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=scomp_link --cov-report=term

# Single file
pytest tests/test_cli.py -v
```

## May the PR be with you

Once submitted, a maintainer will review your PR. CI must be green before merging. If changes are requested, push new commits — don't force-push.

Thank you for making scomp-link better. ⭐
