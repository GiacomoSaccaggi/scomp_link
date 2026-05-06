# Version-Specific Requirements

## Overview

All dependency management is handled via `pyproject.toml`. The `setup.py` is a minimal shim kept for backward compatibility (editable installs). The `requirements/` directory contains legacy per-version files for manual installation but is **not** the source of truth.

**Supported Python versions**: 3.10, 3.11, 3.12, 3.13

Testing is automated via `tox.ini` across all four versions.

## Core Dependencies by Python Version

All version ranges are defined in `pyproject.toml` under `[project] dependencies`.

### numpy

| Python | Version Range |
|--------|--------------|
| 3.10   | >=1.22.0, <1.27.0 |
| 3.11   | >=1.24.0, <1.27.0 |
| 3.12   | >=1.26.0, <2.0.0 |
| 3.13   | >=1.26.0, <3.0.0 |

### pandas

| Python | Version Range |
|--------|--------------|
| 3.10   | >=1.4.0, <2.2.0 |
| 3.11   | >=2.0.0, <2.2.0 |
| 3.12   | >=2.1.0, <2.3.0 |
| 3.13   | >=2.1.0, <3.0.0 |

### scipy

| Python | Version Range |
|--------|--------------|
| 3.10   | >=1.8.0, <1.12.0 |
| 3.11   | >=1.10.0, <1.12.0 |
| 3.12   | >=1.11.0, <1.14.0 |
| 3.13   | >=1.11.0, <2.0.0 |

### scikit-learn

| Python | Version Range |
|--------|--------------|
| 3.10   | >=1.2.0, <1.4.0 |
| 3.11   | >=1.3.0, <1.5.0 |
| 3.12   | >=1.3.0, <1.6.0 |
| 3.13   | >=1.3.0, <2.0.0 |

### matplotlib

| Python | Version Range |
|--------|--------------|
| 3.10   | >=3.6.0, <3.9.0 |
| 3.11   | >=3.7.0, <3.9.0 |
| 3.12   | >=3.8.0, <3.10.0 |
| 3.13   | >=3.8.0, <4.0.0 |

### plotly

| Python | Version Range |
|--------|--------------|
| All    | >=5.0.0, <6.0.0 |

### seaborn

| Python | Version Range |
|--------|--------------|
| All    | >=0.11.0, <0.14.0 |

## NLP Dependencies

### torch

| Python | Version Range |
|--------|--------------|
| 3.10   | >=1.13.0, <2.3.0 |
| 3.11   | >=2.0.0, <2.4.0 |
| 3.12   | >=2.0.0, <2.5.0 |
| 3.13   | >=2.0.0, <3.0.0 |

### transformers

| Python | Version Range |
|--------|--------------|
| All    | >=4.30.0, <5.0.0 |

### spacy

| Python | Version Range |
|--------|--------------|
| All    | >=3.5.0, <4.0.0 |

### faiss-cpu

| Python | Version Range |
|--------|--------------|
| All    | >=1.7.0, <2.0.0 |

### sentence-transformers

| Python | Version Range |
|--------|--------------|
| All    | >=2.2.0, <3.0.0 |

## Computer Vision Dependencies

### tensorflow

| Python | Version Range |
|--------|--------------|
| 3.10   | >=2.10.0, <2.16.0 |
| 3.11   | >=2.12.0, <2.16.0 |
| 3.12   | >=2.13.0, <2.18.0 |
| 3.13   | >=2.13.0, <3.0.0 |

### pillow

| Python | Version Range |
|--------|--------------|
| 3.10–3.11 | >=9.0.0, <11.0.0 |
| 3.12+     | >=10.0.0, <11.0.0 |

## Anomaly Detection Dependencies

| Package | Version Range |
|---------|--------------|
| pytorch-tabnet | >=4.0.0, <5.0.0 |
| statsmodels    | >=0.13.0, <1.0.0 |

## Utility Dependencies

| Package | Version Range |
|---------|--------------|
| tqdm    | >=4.50.0, <5.0.0 |
| PyJWT   | >=2.0.0, <3.0.0 |
| markdown | >=3.3.0, <4.0.0 |
| weasyprint | >=57.0, <63.0 |
| playwright | >=1.40.0, <2.0.0 |

## Installation

### Recommended (from PyPI)

```bash
pip install scomp-link
```

### Development mode (from source)

```bash
git clone https://github.com/GiacomoSaccaggi/scomp-link.git
cd scomp_link
pip install -e .

# With test tools
pip install -e ".[dev]"
```

### Manual (legacy requirements files)

```bash
# Core dependencies for your Python version
pip install -r requirements/requirements-py312.txt

# NLP extras
pip install -r requirements/requirements-nlp-py312.txt

# Image/CV extras
pip install -r requirements/requirements-img-py312.txt

# Utilities
pip install -r requirements/requirements-utils.txt
```

## Testing

```bash
# Run full multi-version test suite
rm -rf .tox && ./run_tox_all_versions.sh

# Run tests only (current env)
pytest tests/ -v

# Run all examples (current env)
bash run_all_examples.sh
```

## File Roles

| File | Purpose |
|------|---------|
| `pyproject.toml` | **Source of truth** — all deps, build config, tool config |
| `setup.py` | Minimal shim for `pip install -e .` backward compat |
| `tox.ini` | Multi-version test orchestration |
| `requirements/` | Legacy per-version files for manual install |
| `requirements.txt` | Quick-start core deps (not version-pinned) |

## Notes

- Python 3.7–3.9 are **no longer supported**. Legacy files in `requirements/` are kept but untested.
- All version constraints are tested via tox on macOS (Apple Silicon) with pyenv-managed interpreters.
- TensorFlow on M1/M2 Macs may show a warning about `tf.keras.optimizers.Adam` running slowly — this is cosmetic and does not affect correctness.

---

📦 [scomp-link on PyPI](https://pypi.org/project/scomp-link/)
