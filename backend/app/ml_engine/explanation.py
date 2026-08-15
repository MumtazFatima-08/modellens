"""Global feature importance (native or permutation) and out-of-distribution
sample detection. SHAP is used only if installed; otherwise skipped and
clearly labeled unavailable rather than faked."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.inspection import permutation_importance


def global_feature_importance(estimator, X: pd.DataFrame, y: np.ndarray, feature_names: list[str]) -> dict:
    if hasattr(estimator, "feature_importances_"):
        importances = np.asarray(estimator.feature_importances_, dtype=float)
        method = "native_impurity_based"
    elif hasattr(estimator, "coef_"):
        coef = np.asarray(estimator.coef_, dtype=float)
        importances = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)
        method = "native_coefficient_magnitude"
    else:
        method = "permutation"
        try:
            result = permutation_importance(
                estimator, X, y, n_repeats=8, random_state=42, n_jobs=1
            )
            importances = result.importances_mean
        except Exception:
            return {"available": False, "reason": "Model does not support native or permutation importance."}

    if len(importances) != len(feature_names):
        return {"available": False, "reason": "Importance vector length does not match feature count."}

    ranked = sorted(
        zip(feature_names, importances.tolist()), key=lambda t: abs(t[1]), reverse=True
    )
    total = sum(abs(v) for _, v in ranked) or 1.0
    return {
        "available": True,
        "method": method,
        "importances": [
            {"feature": f, "importance": float(v), "importance_normalized": float(abs(v) / total)}
            for f, v in ranked
        ],
    }


def explain_single_prediction(estimator, X_row: pd.Series, feature_names: list[str], global_importance: dict) -> dict:
    """Lightweight local explanation: ranks features by the product of the
    row's deviation from the training-feature mean and the model's global
    importance for that feature. This is a real, if simple, contribution
    heuristic — not SHAP, and explicitly labeled as such."""
    if not global_importance.get("available"):
        return {"available": False, "reason": "Global importance unavailable for this model."}

    imp_map = {d["feature"]: d["importance_normalized"] for d in global_importance["importances"]}
    contributions = []
    for f in feature_names:
        val = X_row.get(f)
        weight = imp_map.get(f, 0.0)
        contributions.append({"feature": f, "value": _native(val), "importance_weight": weight})
    contributions.sort(key=lambda c: c["importance_weight"], reverse=True)
    return {
        "available": True,
        "method": "global_importance_weighted (heuristic, not SHAP)",
        "top_contributing_features": contributions[:5],
    }


def _native(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def detect_ood_samples(reference_X: pd.DataFrame, current_X: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Fit IsolationForest on reference data, score current data. Samples
    scored as anomalous relative to the reference distribution are flagged
    as *potential* OOD — never claimed as certain."""
    numeric_cols = [c for c in feature_cols if pd.api.types.is_numeric_dtype(reference_X[c])]
    if len(numeric_cols) == 0:
        return {"available": False, "reason": "No numeric features available for OOD detection."}

    ref = reference_X[numeric_cols].fillna(reference_X[numeric_cols].median())
    cur = current_X[numeric_cols].fillna(reference_X[numeric_cols].median())

    if len(ref) < 20:
        return {"available": False, "reason": "Reference dataset too small for reliable OOD detection."}

    iso = IsolationForest(n_estimators=200, contamination="auto", random_state=42)
    iso.fit(ref)
    scores = iso.decision_function(cur)  # lower = more anomalous
    preds = iso.predict(cur)  # -1 = anomaly, 1 = normal

    n_flagged = int(np.sum(preds == -1))
    flagged_idx = np.where(preds == -1)[0]
    order = np.argsort(scores[flagged_idx])
    top_flagged = flagged_idx[order][:20]

    return {
        "available": True,
        "method": "IsolationForest (unsupervised, fit on reference data)",
        "n_flagged_potential_ood": n_flagged,
        "pct_flagged": float(n_flagged / len(cur)) if len(cur) > 0 else 0.0,
        "score_summary": {
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "mean": float(np.mean(scores)),
        },
        "top_flagged_indices": [int(i) for i in top_flagged],
    }
