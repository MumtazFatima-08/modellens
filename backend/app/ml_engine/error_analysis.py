"""Error analysis: inspect individual failures and detect real statistical
associations between features and error rate (not hardcoded rules)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def build_error_table(
    X: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    task_type: str,
    proba: np.ndarray | None,
    classes: list | None,
) -> pd.DataFrame:
    df = X.copy().reset_index(drop=True)
    df["_true"] = np.asarray(y_true)
    df["_pred"] = np.asarray(y_pred)

    if task_type == "classification":
        df["_is_error"] = (df["_true"] != df["_pred"]).astype(int)
        if proba is not None and classes is not None:
            class_to_idx = {c: i for i, c in enumerate(classes)}
            conf = np.array(
                [proba[i, class_to_idx.get(pred, 0)] for i, pred in enumerate(df["_pred"])]
            )
            df["_confidence"] = conf
        else:
            df["_confidence"] = np.nan
    else:
        residual = df["_true"].astype(float) - df["_pred"].astype(float)
        df["_residual"] = residual
        df["_abs_error"] = residual.abs()
        threshold = df["_abs_error"].quantile(0.90)
        df["_is_error"] = (df["_abs_error"] > threshold).astype(int)

    return df


def top_error_examples(err_df: pd.DataFrame, task_type: str, n: int = 25) -> list[dict]:
    errors = err_df[err_df["_is_error"] == 1].copy()
    if task_type == "classification" and "_confidence" in errors.columns and errors["_confidence"].notna().any():
        errors = errors.sort_values("_confidence", ascending=False)
    elif task_type == "regression" and "_abs_error" in errors.columns:
        errors = errors.sort_values("_abs_error", ascending=False)
    errors = errors.head(n)

    feature_cols = [c for c in err_df.columns if not c.startswith("_")]

    # Population stats (whole dataset, not just errors) — used to say
    # *why* an error looks unusual, in plain terms, per example.
    numeric_stats = {}
    for c in feature_cols:
        if pd.api.types.is_numeric_dtype(err_df[c]):
            mean = err_df[c].mean()
            std = err_df[c].std()
            if std and not np.isnan(std) and std > 0:
                numeric_stats[c] = (mean, std)

    examples = []
    for _, row in errors.iterrows():
        rec = {
            "features": {c: _to_native(row[c]) for c in feature_cols},
            "true_label": _to_native(row["_true"]),
            "predicted_label": _to_native(row["_pred"]),
        }
        if task_type == "classification":
            rec["confidence"] = None if pd.isna(row.get("_confidence")) else float(row["_confidence"])
        else:
            rec["residual"] = float(row["_residual"])
            rec["abs_error"] = float(row["_abs_error"])

        rec["explanation"] = _explain_example(row, numeric_stats, task_type, rec)
        examples.append(rec)
    return examples


def _explain_example(row: pd.Series, numeric_stats: dict, task_type: str, rec: dict) -> str:
    """Plain-English summary of *why this example looks like an outlier* —
    which features are unusually far from the dataset's typical value for
    this row (measured in standard deviations). This describes what makes
    the error's inputs unusual; it is not a causal explanation of the
    model's internal decision."""
    deviations = []
    for feat, (mean, std) in numeric_stats.items():
        val = row.get(feat)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            continue
        z = (val - mean) / std
        if abs(z) >= 1.5:
            direction = "unusually high" if z > 0 else "unusually low"
            deviations.append((abs(z), f"{feat} is {direction} ({z:+.1f} std from typical)"))

    deviations.sort(key=lambda d: -d[0])
    top_devs = [d[1] for d in deviations[:3]]

    if task_type == "classification":
        conf = rec.get("confidence")
        base = f"True label was {rec['true_label']}, model predicted {rec['predicted_label']}"
        if conf is not None:
            base += f" with {conf*100:.1f}% confidence"
        base += "."
    else:
        base = f"True value {rec['true_label']}, predicted {rec['predicted_label']} (error of {rec.get('abs_error', 0):.3g})."

    if top_devs:
        return base + " Unusual inputs for this row: " + "; ".join(top_devs) + "."
    return base + " No individual feature stood out as statistically unusual (>1.5 std) versus the rest of the dataset — this may be a harder boundary case rather than an outlier."


def _to_native(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def error_concentration(err_df: pd.DataFrame, feature_cols: list[str], min_group_size: int = 20) -> list[dict]:
    """For each feature, test whether error rate differs meaningfully between
    a high-value group and the rest (numeric: top quartile vs rest, via
    two-proportion z-test; categorical: chi-square across categories).
    Only real, computed associations — no hardcoded thresholds."""
    overall_error_rate = float(err_df["_is_error"].mean())
    results = []

    for col in feature_cols:
        series = err_df[col]
        try:
            if pd.api.types.is_numeric_dtype(series):
                q75 = series.quantile(0.75)
                high_mask = series > q75
                low_mask = ~high_mask
                n_high, n_low = high_mask.sum(), low_mask.sum()
                if n_high < min_group_size or n_low < min_group_size:
                    continue
                err_high = err_df.loc[high_mask, "_is_error"].mean()
                err_low = err_df.loc[low_mask, "_is_error"].mean()

                count = np.array([
                    err_df.loc[high_mask, "_is_error"].sum(),
                    err_df.loc[low_mask, "_is_error"].sum(),
                ])
                nobs = np.array([n_high, n_low])
                p_pool = count.sum() / nobs.sum()
                if p_pool in (0, 1):
                    continue
                se = np.sqrt(p_pool * (1 - p_pool) * (1 / nobs[0] + 1 / nobs[1]))
                if se == 0:
                    continue
                z = (err_high - err_low) / se
                p_value = float(2 * (1 - stats.norm.cdf(abs(z))))

                results.append(
                    {
                        "feature": col,
                        "condition": f"{col} > {round(float(q75), 4)} (top quartile)",
                        "group_error_rate": float(err_high),
                        "rest_error_rate": float(err_low),
                        "overall_error_rate": overall_error_rate,
                        "lift": float(err_high - overall_error_rate),
                        "group_size": int(n_high),
                        "p_value": p_value,
                        "significant": bool(p_value < 0.05),
                    }
                )
            else:
                counts = err_df.groupby(col)["_is_error"].agg(["sum", "count"])
                counts = counts[counts["count"] >= min_group_size]
                if len(counts) < 2:
                    continue
                contingency = np.array(
                    [[row["sum"], row["count"] - row["sum"]] for _, row in counts.iterrows()]
                )
                if contingency.shape[0] < 2 or contingency.sum() == 0:
                    continue
                chi2, p_value, _, _ = stats.chi2_contingency(contingency)
                worst_cat = (counts["sum"] / counts["count"]).idxmax()
                worst_rate = float((counts["sum"] / counts["count"]).max())
                worst_n = int(counts.loc[worst_cat, "count"])
                results.append(
                    {
                        "feature": col,
                        "condition": f"{col} == '{worst_cat}'",
                        "group_error_rate": worst_rate,
                        "rest_error_rate": overall_error_rate,
                        "overall_error_rate": overall_error_rate,
                        "lift": float(worst_rate - overall_error_rate),
                        "group_size": worst_n,
                        "p_value": float(p_value),
                        "significant": bool(p_value < 0.05),
                    }
                )
        except Exception:
            continue

    results.sort(key=lambda r: (not r["significant"], -abs(r["lift"])))
    return results
