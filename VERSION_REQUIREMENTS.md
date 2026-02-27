# Version-Specific Requirements

## Overview
Created Python version-specific requirements files (3.7-3.13) with compatible package versions. The `setup.py` automatically detects the Python version and installs the correct dependencies.

## Directory Structure
```
requirements/
├── requirements-py37.txt      # Python 3.7 compatible versions
├── requirements-py38.txt      # Python 3.8 compatible versions
├── requirements-py39.txt      # Python 3.9 compatible versions
├── requirements-py310.txt     # Python 3.10 compatible versions
├── requirements-py311.txt     # Python 3.11 compatible versions
├── requirements-py312.txt     # Python 3.12 compatible versions
├── requirements-py313.txt     # Python 3.13 compatible versions
├── requirements-utils.txt     # Utility packages (all versions)
├── requirements-nlp-py37.txt  # NLP packages for Python 3.7
├── requirements-nlp-py38.txt  # NLP packages for Python 3.8-3.9
├── requirements-nlp-py310.txt # NLP packages for Python 3.10+
├── requirements-img-py37.txt  # Image/CV packages for Python 3.7-3.9
└── requirements-img-py310.txt # Image/CV packages for Python 3.10+
```

## Core Dependencies by Python Version

### Python 3.7 (Last supported versions)
- numpy: 1.16.0 - 1.21.x
- pandas: 1.0.0 - 1.3.x
- scipy: 1.3.0 - 1.7.x
- scikit-learn: 0.22.0 - 1.0.x
- matplotlib: 3.0.0 - 3.5.x
- plotly: 4.0.0 - 5.14.x

### Python 3.8
- numpy: 1.17.0 - 1.24.x
- pandas: 1.0.0 - 2.0.x
- scipy: 1.4.0 - 1.10.x
- scikit-learn: 0.23.0 - 1.2.x
- matplotlib: 3.1.0 - 3.7.x
- plotly: 4.5.0 - 5.17.x

### Python 3.9
- numpy: 1.19.0 - 1.26.x
- pandas: 1.2.0 - 2.1.x
- scipy: 1.5.0 - 1.11.x
- scikit-learn: 0.24.0 - 1.3.x
- matplotlib: 3.3.0 - 3.8.x
- plotly: 4.14.0 - 5.19.x

### Python 3.10
- numpy: 1.21.0 - 1.x
- pandas: 1.3.0 - 2.2.x
- scipy: 1.7.0 - 1.12.x
- scikit-learn: 1.0.0 - 1.4.x
- matplotlib: 3.4.0 - 3.9.x
- plotly: 5.0.0 - 5.x

### Python 3.11
- numpy: 1.23.0 - 1.x
- pandas: 1.5.0 - 2.2.x
- scipy: 1.9.0 - 1.13.x
- scikit-learn: 1.1.0 - 1.5.x
- matplotlib: 3.6.0 - 3.9.x
- plotly: 5.10.0 - 5.x

### Python 3.12
- numpy: 1.26.0 - 1.x
- pandas: 2.1.0 - 2.2.x
- scipy: 1.11.0 - 1.13.x
- scikit-learn: 1.3.0 - 1.5.x
- matplotlib: 3.8.0 - 3.9.x
- plotly: 5.17.0 - 5.x

### Python 3.13
- numpy: 1.26.0 - 2.x
- pandas: 2.1.0 - 2.x
- scipy: 1.11.0 - 1.x
- scikit-learn: 1.3.0 - 1.x
- matplotlib: 3.8.0 - 3.x
- plotly: 5.17.0 - 5.x

## Optional Dependencies

### NLP (Natural Language Processing)
**Python 3.7:**
- torch: 1.7.0 - 1.13.x
- transformers: 4.0.0 - 4.29.x
- spacy: 3.0.0 - 3.4.x

**Python 3.8-3.9:**
- torch: 1.8.0 - 2.1.x
- transformers: 4.10.0 - 4.39.x
- spacy: 3.2.0 - 3.6.x

**Python 3.10+:**
- torch: 2.0.0 - 2.4.x
- transformers: 4.30.0 - 4.x
- spacy: 3.5.0 - 3.x

### Image/Computer Vision
**Python 3.7-3.9:**
- tensorflow: 2.4.0 - 2.12.x
- pillow: 8.0.0 - 9.x

**Python 3.10+:**
- tensorflow: 2.13.0 - 2.17.x
- pillow: 10.0.0 - 10.x

### Utilities (All versions)
- tqdm: 4.50.0 - 4.x
- PyJWT: 2.0.0 - 2.x

## Installation Methods

### Method 1: Automatic (Recommended)
The `setup.py` automatically detects your Python version and installs compatible packages:

```bash
# Install core dependencies only
pip install .

# Install with NLP support
pip install .[nlp]

# Install with Image/CV support
pip install .[img]

# Install with utilities
pip install .[utils]

# Install everything
pip install .[all]
```

### Method 2: Manual
Install specific requirements file for your Python version:

```bash
# Core dependencies
pip install -r requirements/requirements-py39.txt

# Optional: NLP
pip install -r requirements/requirements-nlp-py38.txt

# Optional: Image/CV
pip install -r requirements/requirements-img-py37.txt

# Optional: Utilities
pip install -r requirements/requirements-utils.txt
```

### Method 3: Development Mode
Install in editable mode for development:

```bash
pip install -e .[all]
```

## How setup.py Works

The `setup.py` file:
1. Detects the current Python version using `sys.version_info`
2. Selects the appropriate requirements file from `requirements/` directory
3. Loads dependencies from the selected file
4. Installs version-compatible packages

Example logic:
```python
if py_version >= (3, 13):
    req_file = "requirements/requirements-py313.txt"
elif py_version >= (3, 12):
    req_file = "requirements/requirements-py312.txt"
# ... etc
```

## Testing Installation

Verify correct installation:

```bash
# Check Python version
python --version

# Install package
pip install .

# Verify imports
python -c "import numpy, pandas, scipy, sklearn, matplotlib, plotly; print('✅ All core packages installed')"

# Test with NLP (if installed)
python -c "import torch, transformers, spacy; print('✅ NLP packages installed')"

# Test with Image/CV (if installed)
python -c "import tensorflow, PIL; print('✅ Image packages installed')"
```

## Version Compatibility Notes

### Breaking Changes
- **NumPy 1.20+**: Requires Python 3.7+
- **NumPy 1.22+**: Dropped Python 3.7 support
- **Pandas 2.0+**: Requires Python 3.8+
- **TensorFlow 2.11+**: Dropped Python 3.7 support
- **PyTorch 2.0+**: Requires Python 3.8+

### Recommended Versions
- **Python 3.9-3.11**: Best compatibility with all packages
- **Python 3.12+**: Latest features, some packages may lag
- **Python 3.7**: Limited to older package versions (EOL)

## Troubleshooting

### Issue: Package version conflicts
**Solution**: Use the version-specific requirements file for your Python version

### Issue: setup.py fails to find requirements file
**Solution**: Ensure you're running setup.py from the project root directory

### Issue: Optional dependencies fail to install
**Solution**: Check that you're using compatible Python version for that feature (e.g., NLP requires 3.7+, latest TensorFlow requires 3.8+)

## Maintenance

When updating package versions:
1. Test compatibility with each Python version
2. Update corresponding `requirements-py<version>.txt` file
3. Verify installation with `pip install .`
4. Run test suite: `pytest tests/`
