"""Data drift detection: real distribution-comparison statistics between a
reference dataset and current dataset. Method chosen per feature type."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _psi_numeric(ref: pd.Series, cur: pd.Series, bins: int = 10) -> float:
    ref = ref.dropna().astype(float)
    cur = cur.dropna().astype(float)
    if len(ref) == 0 or len(cur) == 0:
        return float("nan")
    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(ref, quantiles))
    if len(edges) < 3:
        return 0.0
    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_pct = np.clip(ref_counts / max(ref_counts.sum(), 1), 1e-6, None)
    cur_pct = np.clip(cur_counts / max(cur_counts.sum(), 1), 1e-6, None)
    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


def _psi_categorical(ref: pd.Series, cur: pd.Series) -> float:
    ref = ref.dropna()
    cur = cur.dropna()
    categories = set(ref.unique()) | set(cur.unique())
    ref_counts = ref.value_counts()
    cur_counts = cur.value_counts()
    psi = 0.0
    for cat in categories:
        ref_pct = max(ref_counts.get(cat, 0) / max(len(ref), 1), 1e-6)
        cur_pct = max(cur_counts.get(cat, 0) / max(len(cur), 1), 1e-6)
        psi += (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)
    return float(psi)


def _status_from_psi(psi: float) -> str:
    if np.isnan(psi):
        return "unknown"
    if psi < 0.1:
        return "normal"
    if psi < 0.25:
        return "drift"
    return "severe_drift"


def analyze_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame, feature_cols: list[str]) -> list[dict]:
    results = []
    for col in feature_cols:
        if col not in reference_df.columns or col not in current_df.columns:
            continue
        ref_series = reference_df[col]
        cur_series = current_df[col]

        if pd.api.types.is_numeric_dtype(ref_series):
            psi = _psi_numeric(ref_series, cur_series)
            try:
                ks_stat, ks_p = stats.ks_2samp(
                    ref_series.dropna().astype(float), cur_series.dropna().astype(float)
                )
            except Exception:
                ks_stat, ks_p = float("nan"), float("nan")
            try:
                wasserstein = float(
                    stats.wasserstein_distance(
                        ref_series.dropna().astype(float), cur_series.dropna().astype(float)
                    )
                )
            except Exception:
                wasserstein = float("nan")

            results.append(
                {
                    "feature": col,
                    "feature_type": "numeric",
                    "psi": psi,
                    "ks_statistic": float(ks_stat),
                    "ks_p_value": float(ks_p),
                    "wasserstein_distance": wasserstein,
                    "status": _status_from_psi(psi),
                    "reference_mean": float(ref_series.mean()) if ref_series.notna().any() else None,
                    "current_mean": float(cur_series.mean()) if cur_series.notna().any() else None,
                }
            )
        else:
            psi = _psi_categorical(ref_series, cur_series)
            try:
                categories = sorted(set(ref_series.dropna().unique()) | set(cur_series.dropna().unique()), key=str)
                ref_counts = [int((ref_series == c).sum()) for c in categories]
                cur_counts = [int((cur_series == c).sum()) for c in categories]
                chi2, chi_p, _, _ = stats.chi2_contingency([ref_counts, cur_counts])
            except Exception:
                chi2, chi_p = float("nan"), float("nan")

            results.append(
                {
                    "feature": col,
                    "feature_type": "categorical",
                    "psi": psi,
                    "chi2_statistic": float(chi2),
                    "chi2_p_value": float(chi_p),
                    "status": _status_from_psi(psi),
                    "reference_top_category": str(ref_series.mode().iloc[0]) if not ref_series.mode().empty else None,
                    "current_top_category": str(cur_series.mode().iloc[0]) if not cur_series.mode().empty else None,
                }
            )

    results.sort(key=lambda r: (r["psi"] if not np.isnan(r["psi"]) else -1), reverse=True)
    return results
