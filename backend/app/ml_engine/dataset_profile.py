"""Dataset profiling: real column-level statistics so a user can see what's
actually in their data before/after running an investigation. No guessing —
every number here comes from the actual uploaded dataframe."""
from __future__ import annotations

import numpy as np
import pandas as pd


def profile_dataset(df: pd.DataFrame, target_column: str | None = None) -> dict:
    n_rows = int(len(df))
    columns = []

    for col in df.columns:
        series = df[col]
        n_missing = int(series.isna().sum())
        pct_missing = float(n_missing / n_rows) if n_rows else 0.0
        is_numeric = pd.api.types.is_numeric_dtype(series)

        entry = {
            "name": col,
            "is_target": col == target_column,
            "dtype": "numeric" if is_numeric else "categorical",
            "n_missing": n_missing,
            "pct_missing": pct_missing,
            "n_unique": int(series.nunique(dropna=True)),
        }

        if is_numeric:
            clean = series.dropna().astype(float)
            if len(clean) > 0:
                entry.update(
                    {
                        "min": float(clean.min()),
                        "max": float(clean.max()),
                        "mean": float(clean.mean()),
                        "median": float(clean.median()),
                        "std": float(clean.std()) if len(clean) > 1 else 0.0,
                    }
                )
        else:
            top = series.dropna().astype(str).value_counts().head(5)
            entry["top_values"] = [
                {"value": v, "count": int(c), "pct": float(c / n_rows) if n_rows else 0.0}
                for v, c in top.items()
            ]

        columns.append(entry)

    target_summary = None
    if target_column and target_column in df.columns:
        t = df[target_column]
        if pd.api.types.is_numeric_dtype(t) and t.nunique(dropna=True) > 20:
            clean = t.dropna().astype(float)
            target_summary = {
                "type": "continuous",
                "min": float(clean.min()),
                "max": float(clean.max()),
                "mean": float(clean.mean()),
                "std": float(clean.std()) if len(clean) > 1 else 0.0,
            }
        else:
            counts = t.value_counts(dropna=True)
            total = int(counts.sum())
            target_summary = {
                "type": "categorical",
                "class_counts": {str(k): int(v) for k, v in counts.items()},
                "class_balance_pct": {str(k): float(v / total) if total else 0.0 for k, v in counts.items()},
                "is_imbalanced": bool((counts.min() / counts.max()) < 0.2) if len(counts) > 1 and counts.max() > 0 else False,
            }

    return {
        "n_rows": n_rows,
        "n_columns": int(len(df.columns)),
        "columns": columns,
        "target_summary": target_summary,
        "duplicate_rows": int(df.duplicated().sum()),
    }
