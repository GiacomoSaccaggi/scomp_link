# -*- coding: utf-8 -*-
"""
██████╗ ███████╗██████╗ ███████╗██╗███████╗████████╗
██╔══██╗██╔════╝██╔══██╗██╔════╝██║██╔════╝╚══██╔══╝
██████╔╝█████╗  ██████╔╝███████╗██║███████╗   ██║
██╔═══╝ ██╔══╝  ██╔══██╗╚════██║██║╚════██║   ██║
██║     ███████╗██║  ██║███████║██║███████║   ██║
╚═╝     ╚══════╝╚═╝  ╚═╝╚══════╝╚═╝╚══════╝   ╚═╝

.scomp format: A self-contained pipeline artifact.

Structure (zip-based):
    artifact.scomp
    ├── manifest.json      # Version, timestamp, python version, package versions, metadata
    ├── model.pkl          # Pickled fitted model
    ├── preprocessor.pkl   # Pickled preprocessor (ColumnTransformer, scaler, etc.)
    ├── config.json        # Pipeline configuration (params, feature names, target, task type)
    ├── metrics.json       # Training/validation metrics
    ├── feature_schema.json # Column names, types, expected ranges
    └── sample_data.parquet # Optional: small sample of training data (for drift detection)
"""

import io
import json
import pickle
import platform
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd

from scomp_link.utils.logger import get_logger

logger = get_logger(__name__)


SCOMP_FORMAT_VERSION = "1.0"
MAGIC_BYTES = b"SCOMP\x01"


class ScompArtifact:
    """
    Serialize and deserialize complete ML pipelines as .scomp files.

    Usage example:
        # Save
        artifact = ScompArtifact()
        artifact.set_model(fitted_model)
        artifact.set_preprocessor(column_transformer)
        artifact.set_config(task_type='regression', target_col='y', feature_cols=['x1','x2'])
        artifact.set_metrics({'rmse': 0.42, 'r2': 0.91})
        artifact.set_sample_data(X_train.head(100))
        artifact.save('my_pipeline.scomp')

        # Load
        loaded = ScompArtifact.load('my_pipeline.scomp')
        model = loaded.model
        preprocessor = loaded.preprocessor
        loaded.predict(new_data)  # preprocessor + model in one call
    """

    def __init__(self):
        self.model = None
        self.preprocessor = None
        self.config: Dict[str, Any] = {}
        self.metrics: Dict[str, Any] = {}
        self.feature_schema: Dict[str, Any] = {}
        self.sample_data: Optional[pd.DataFrame] = None
        self.metadata: Dict[str, Any] = {}

    def set_model(self, model) -> "ScompArtifact":
        """Set the fitted model."""
        self.model = model
        return self

    def set_preprocessor(self, preprocessor) -> "ScompArtifact":
        """Set the preprocessing pipeline (ColumnTransformer, scaler, encoder, etc.)."""
        self.preprocessor = preprocessor
        return self

    def set_config(self, **kwargs) -> "ScompArtifact":
        """Set pipeline configuration (task_type, target_col, feature_cols, etc.)."""
        self.config.update(kwargs)
        return self

    def set_metrics(self, metrics: Dict[str, float]) -> "ScompArtifact":
        """Set training/validation metrics."""
        self.metrics = metrics
        return self

    def set_feature_schema(self, X: pd.DataFrame) -> "ScompArtifact":
        """Infer and store feature schema from a DataFrame."""
        schema = {}
        for col in X.columns:
            info = {"dtype": str(X[col].dtype)}
            if np.issubdtype(X[col].dtype, np.number):
                info["min"] = float(X[col].min())
                info["max"] = float(X[col].max())
                info["mean"] = float(X[col].mean())
            else:
                info["n_unique"] = int(X[col].nunique())
                info["categories"] = X[col].unique().tolist()[:50]  # cap at 50
            schema[col] = info
        self.feature_schema = schema
        return self

    def set_sample_data(self, df: pd.DataFrame, max_rows: int = 100) -> "ScompArtifact":
        """Store a small sample of training data (useful for drift detection later)."""
        self.sample_data = df.head(max_rows)
        return self

    def set_metadata(self, **kwargs) -> "ScompArtifact":
        """Set arbitrary metadata (author, description, experiment_id, etc.)."""
        self.metadata.update(kwargs)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Run preprocessor (if set) + model.predict() in one call."""
        if self.model is None:
            raise ValueError("No model loaded.")
        if self.preprocessor is not None:
            X = self.preprocessor.transform(X)
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Run preprocessor + model.predict_proba() (classification only)."""
        if self.model is None:
            raise ValueError("No model loaded.")
        if self.preprocessor is not None:
            X = self.preprocessor.transform(X)
        return self.model.predict_proba(X)

    def save(self, path: Union[str, Path]) -> Path:
        """Save the artifact as a .scomp file."""
        path = Path(path)
        if not path.suffix:
            path = path.with_suffix(".scomp")

        manifest = {
            "format_version": SCOMP_FORMAT_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "python_version": sys.version,
            "platform": platform.platform(),
            "packages": self._get_package_versions(),
            "has_model": self.model is not None,
            "has_preprocessor": self.preprocessor is not None,
            "has_sample_data": self.sample_data is not None,
            **self.metadata,
        }

        with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
            # Magic bytes for file identification
            zf.writestr("__magic__", MAGIC_BYTES)
            zf.writestr("manifest.json", json.dumps(manifest, indent=2, default=str))
            zf.writestr("config.json", json.dumps(self.config, indent=2, default=str))
            zf.writestr("metrics.json", json.dumps(self.metrics, indent=2, default=str))
            zf.writestr("feature_schema.json", json.dumps(self.feature_schema, indent=2, default=str))

            if self.model is not None:
                zf.writestr("model.pkl", pickle.dumps(self.model))

            if self.preprocessor is not None:
                zf.writestr("preprocessor.pkl", pickle.dumps(self.preprocessor))

            if self.sample_data is not None:
                buf = io.BytesIO()
                self.sample_data.to_parquet(buf, index=False)
                zf.writestr("sample_data.parquet", buf.getvalue())

        logger.info(f"✅ Saved pipeline artifact: {path} ({path.stat().st_size / 1024:.1f} KB)")
        return path

    @classmethod
    def load(cls, path: Union[str, Path]) -> "ScompArtifact":
        """Load a .scomp artifact.

        ⚠️ SECURITY WARNING: .scomp files use pickle internally.
        Only load files you created yourself or received from a trusted source.
        Loading untrusted files can execute arbitrary code.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {path}")

        artifact = cls()

        with zipfile.ZipFile(path, "r") as zf:
            # Verify magic
            magic = zf.read("__magic__")
            if magic != MAGIC_BYTES:
                raise ValueError("Invalid .scomp file: bad magic bytes")

            manifest = json.loads(zf.read("manifest.json"))
            artifact.metadata = {
                k: v
                for k, v in manifest.items()
                if k
                not in (
                    "format_version",
                    "has_model",
                    "has_preprocessor",
                    "has_sample_data",
                    "packages",
                    "python_version",
                    "platform",
                    "created_at",
                )
            }

            artifact.config = json.loads(zf.read("config.json"))
            artifact.metrics = json.loads(zf.read("metrics.json"))
            artifact.feature_schema = json.loads(zf.read("feature_schema.json"))

            if manifest.get("has_model"):
                artifact.model = pickle.loads(zf.read("model.pkl"))

            if manifest.get("has_preprocessor"):
                artifact.preprocessor = pickle.loads(zf.read("preprocessor.pkl"))

            if manifest.get("has_sample_data"):
                buf = io.BytesIO(zf.read("sample_data.parquet"))
                artifact.sample_data = pd.read_parquet(buf)

        logger.info(f"✅ Loaded pipeline artifact: {path} (created: {manifest.get('created_at', '?')})")
        return artifact

    @staticmethod
    def is_scomp_file(path: Union[str, Path]) -> bool:
        """Check if a file is a valid .scomp artifact."""
        try:
            with zipfile.ZipFile(path, "r") as zf:
                return zf.read("__magic__") == MAGIC_BYTES
        except Exception:
            return False

    def info(self) -> Dict[str, Any]:
        """Return a summary of the artifact contents."""
        return {
            "config": self.config,
            "metrics": self.metrics,
            "n_features": len(self.feature_schema),
            "has_model": self.model is not None,
            "has_preprocessor": self.preprocessor is not None,
            "has_sample_data": self.sample_data is not None,
            "model_type": type(self.model).__name__ if self.model else None,
            "metadata": self.metadata,
        }

    @staticmethod
    def _get_package_versions() -> Dict[str, str]:
        """Capture versions of key packages for reproducibility."""
        versions = {}
        for pkg in ["numpy", "pandas", "scikit-learn", "scipy", "polars", "scomp_link"]:
            try:
                mod = __import__(pkg.replace("-", "_"))
                versions[pkg] = getattr(mod, "__version__", "?")
            except ImportError:
                pass
        return versions


if __name__ == "__main__":
    # Sample data
    import os
    import tempfile

    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split

    np.random.seed(42)
    size_df = 300
    X = pd.DataFrame(
        {
            "x1": np.random.randn(size_df),
            "x2": np.random.randn(size_df),
            "x3": np.random.randn(size_df),
        }
    )
    y = 2 * X["x1"] + 0.5 * X["x2"] + np.random.randn(size_df) * 0.1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X_train, y_train)

    # Test save
    logger.info("🔬 Testing ScompArtifact save...")
    artifact = ScompArtifact()
    artifact.set_model(model)
    artifact.set_config(task_type="regression", target_col="y", feature_cols=["x1", "x2", "x3"])
    artifact.set_metrics({"r2": 0.95, "rmse": 0.12})
    artifact.set_feature_schema(X_train)
    artifact.set_sample_data(X_train)
    artifact.set_metadata(author="test", description="Demo pipeline")

    fd, path = tempfile.mkstemp(suffix=".scomp")
    os.close(fd)
    artifact.save(path)

    # Test load
    logger.info("\n🔬 Testing ScompArtifact load...")
    loaded = ScompArtifact.load(path)
    preds = loaded.predict(X_test)
    logger.info(f"✅ Predictions shape: {preds.shape}")
    logger.info(f"✅ Info: {loaded.info()}")
    logger.info(f"✅ Is scomp file: {ScompArtifact.is_scomp_file(path)}")

    os.unlink(path)
