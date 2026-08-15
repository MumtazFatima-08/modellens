"""Reads tabular datasets regardless of format (.csv, .xlsx, .xls) behind
one function, so the rest of the engine never has to branch on file type."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


SUPPORTED_DATASET_EXTENSIONS = {".csv", ".xlsx", ".xls"}


class UnsupportedDatasetError(Exception):
    pass


def read_tabular(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix in (".xlsx", ".xls"):
            return pd.read_excel(path)
    except Exception as e:
        raise UnsupportedDatasetError(f"Could not parse '{path.name}': {e}") from e
    raise UnsupportedDatasetError(
        f"Unsupported dataset file extension '{suffix}'. Expected one of: "
        f"{', '.join(sorted(SUPPORTED_DATASET_EXTENSIONS))}."
    )
