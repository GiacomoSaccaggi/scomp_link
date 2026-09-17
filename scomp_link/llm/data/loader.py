# -*- coding: utf-8 -*-
"""
██╗      ██████╗  █████╗ ██████╗ ███████╗██████╗
██║     ██╔═══██╗██╔══██╗██╔══██╗██╔════╝██╔══██╗
██║     ██║   ██║███████║██║  ██║█████╗  ██████╔╝
██║     ██║   ██║██╔══██║██║  ██║██╔══╝  ██╔══██╗
███████╗╚██████╔╝██║  ██║██████╔╝███████╗██║  ██║
╚══════╝ ╚═════╝ ╚═╝  ╚═╝╚═════╝ ╚══════╝╚═╝  ╚═╝

Load datasets from CSV, JSON, JSONL, Parquet, DataFrames, or HF Dataset objects.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Union

from scomp_link.exceptions import DataValidationError

if TYPE_CHECKING:
    import pandas as pd
    from datasets import Dataset

_SUPPORTED_FORMATS = ("csv", "json", "jsonl", "parquet")


def load_dataset(
    source: str | Path | pd.DataFrame | Dataset,
    text_field: str = "text",
) -> Dataset:
    import pandas as pd

    try:
        from datasets import Dataset
    except ImportError:
        raise ImportError("load_dataset requires the 'datasets' package. " "Install with: pip install scomp-link[llm]")

    if isinstance(source, Dataset):
        ds = source
    elif isinstance(source, pd.DataFrame):
        ds = _from_dataframe(source, Dataset)
    elif isinstance(source, (str, Path)):
        ds = _from_path(Path(source), Dataset, pd)
    else:
        raise DataValidationError(
            f"Unsupported dataset type: {type(source).__name__}. "
            f"Supported: str/Path (.{', .'.join(_SUPPORTED_FORMATS)}), "
            f"pd.DataFrame, or HuggingFace Dataset"
        )

    _validate(ds, text_field)
    return ds


def _from_dataframe(df: pd.DataFrame, dataset_cls: type) -> Dataset:
    if len(df) == 0:
        raise DataValidationError("Dataset is empty (zero rows)")
    return dataset_cls.from_pandas(df)


def _from_path(path: Path, dataset_cls: type, pd_module: Any) -> Dataset:
    suffix = path.suffix.lower().lstrip(".")
    if suffix not in _SUPPORTED_FORMATS:
        raise DataValidationError(
            f"Unsupported file format: '.{suffix}'. " f"Supported formats: {', '.join(_SUPPORTED_FORMATS)}"
        )

    if not path.exists():
        raise DataValidationError(f"File not found: {path}")

    if suffix == "csv":
        df = pd_module.read_csv(path)
    elif suffix in ("json", "jsonl"):
        df = pd_module.read_json(path, lines=(suffix == "jsonl"))
    elif suffix == "parquet":
        df = pd_module.read_parquet(path)

    if len(df) == 0:
        raise DataValidationError(f"Dataset is empty (zero rows): {path}")

    return dataset_cls.from_pandas(df)


def _validate(ds: Dataset, text_field: str) -> None:
    if len(ds) == 0:
        raise DataValidationError("Dataset is empty (zero rows)")
    if text_field not in ds.column_names:
        raise DataValidationError(
            f"Dataset is missing required column '{text_field}'. " f"Available columns: {ds.column_names}"
        )


if __name__ == "__main__":
    import json
    import tempfile
    from pathlib import Path

    # Create a tiny CSV and load it
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write("text,label\nhello world,pos\ngoodbye world,neg\nhi there,pos\n")
        csv_path = f.name

    ds = load_dataset(csv_path, text_field="text")
    print(f"Loaded CSV: {len(ds)} rows, columns={ds.column_names}")
    print(f"First row: {ds[0]}")
    Path(csv_path).unlink()

    # Try loading something bad
    try:
        load_dataset(csv_path, text_field="missing_col")
    except Exception as e:
        print(f"Expected error: {type(e).__name__}: {e}")
