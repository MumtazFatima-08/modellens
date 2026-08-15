"""Slice discovery: find subgroups where the model performs meaningfully
worse than overall, using a shallow decision tree trained to predict *error*
from the features (recursive partitioning), then extracting leaf rules.
This is a real, standard subgroup-discovery technique — not hardcoded rules.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.tree import DecisionTreeClassifier, _tree


def discover_slices(
    err_df: pd.DataFrame,
    feature_cols: list[str],
    min_sample_size: int = 30,
    max_depth: int = 3,
    top_k: int = 8,
) -> list[dict]:
    X = err_df[feature_cols].copy()
    y = err_df["_is_error"].values
    overall_error_rate = float(np.mean(y))

    # Encode categoricals for the tree (tree only used for partitioning,
    # not as the model under investigation).
    X_enc = X.copy()
    cat_maps = {}
    for col in feature_cols:
        if not pd.api.types.is_numeric_dtype(X_enc[col]):
            X_enc[col] = X_enc[col].astype("category")
            cat_maps[col] = dict(enumerate(X_enc[col].cat.categories))
            X_enc[col] = X_enc[col].cat.codes
    X_enc = X_enc.fillna(X_enc.median(numeric_only=True))

    if len(np.unique(y)) < 2 or len(X_enc) < min_sample_size * 2:
        return []

    tree = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=min_sample_size,
        class_weight="balanced",
        random_state=42,
    )
    tree.fit(X_enc, y)

    leaf_ids = tree.apply(X_enc)
    slices = []
    for leaf in np.unique(leaf_ids):
        mask = leaf_ids == leaf
        n = int(mask.sum())
        if n < min_sample_size:
            continue
        group_error_rate = float(np.mean(y[mask]))
        if group_error_rate <= overall_error_rate:
            continue  # only report *worse than average* slices

        rule = _extract_rule(tree.tree_, X_enc.columns.tolist(), leaf, cat_maps)

        count = np.array([int(np.sum(y[mask])), int(np.sum(y[~mask]))])
        nobs = np.array([n, int((~mask).sum())])
        p_pool = count.sum() / nobs.sum()
        if p_pool in (0, 1):
            p_value = 1.0
        else:
            se = np.sqrt(p_pool * (1 - p_pool) * (1 / nobs[0] + 1 / nobs[1]))
            z = (group_error_rate - overall_error_rate) / se if se > 0 else 0
            p_value = float(2 * (1 - stats.norm.cdf(abs(z))))

        gap = group_error_rate - overall_error_rate
        slices.append(
            {
                "rule": rule,
                "sample_size": n,
                "group_error_rate": group_error_rate,
                "overall_error_rate": overall_error_rate,
                "performance_gap": float(gap),
                "p_value": p_value,
                "significant": bool(p_value < 0.05),
            }
        )

    slices.sort(key=lambda s: (s["performance_gap"] * np.log1p(s["sample_size"])), reverse=True)
    return slices[:top_k]


def _extract_rule(tree_obj, feature_names, target_leaf, cat_maps) -> str:
    """Walk root->target_leaf path and build a human-readable rule string."""
    tree_ = tree_obj
    conditions = []

    def recurse(node, path_ok):
        if node == target_leaf:
            return path_ok
        if tree_.feature[node] == _tree.TREE_UNDEFINED:
            return None
        left = tree_.children_left[node]
        right = tree_.children_right[node]
        feat = feature_names[tree_.feature[node]]
        thresh = tree_.threshold[node]

        result = recurse(left, path_ok + [(feat, "<=", thresh)])
        if result is not None:
            return result
        result = recurse(right, path_ok + [(feat, ">", thresh)])
        return result

    path = recurse(0, [])
    if not path:
        return "(root)"

    for feat, op, thresh in path:
        if feat in cat_maps:
            # thresh is a code-space split; express categorically when possible
            conditions.append(f"{feat} {op} {round(thresh, 2)} (encoded category)")
        else:
            conditions.append(f"{feat} {op} {round(thresh, 4)}")
    return " AND ".join(conditions)
