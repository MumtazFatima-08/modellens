"""Model evaluation: real, task-appropriate metrics only."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    adjusted_rand_score,
    average_precision_score,
    calinski_harabasz_score,
    confusion_matrix,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    normalized_mutual_info_score,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)


def evaluate_classification(
    y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray | None, classes: list[Any]
) -> dict:
    n_classes = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    accuracy = float(accuracy_score(y_true, y_pred))
    precision_macro = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    recall_macro = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    precision_weighted = float(precision_score(y_true, y_pred, average="weighted", zero_division=0))
    recall_weighted = float(recall_score(y_true, y_pred, average="weighted", zero_division=0))
    f1_weighted = float(f1_score(y_true, y_pred, average="weighted", zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=classes)

    per_class = []
    prec_pc = precision_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    rec_pc = recall_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    f1_pc = f1_score(y_true, y_pred, labels=classes, average=None, zero_division=0)
    for i, c in enumerate(classes):
        support = int(np.sum(np.asarray(y_true) == c))
        per_class.append(
            {
                "class": str(c),
                "precision": float(prec_pc[i]),
                "recall": float(rec_pc[i]),
                "f1": float(f1_pc[i]),
                "support": support,
            }
        )

    roc_auc = None
    pr_auc = None
    if proba is not None:
        try:
            if n_classes == 2:
                pos_idx = class_to_idx[classes[1]]
                roc_auc = float(roc_auc_score(y_true, proba[:, pos_idx]))
                y_bin = (np.asarray(y_true) == classes[1]).astype(int)
                pr_auc = float(average_precision_score(y_bin, proba[:, pos_idx]))
            elif n_classes > 2:
                roc_auc = float(
                    roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=classes)
                )
        except Exception:
            # ROC-AUC/PR-AUC undefined for degenerate cases (e.g. a class
            # missing from y_true entirely) — omit rather than fabricate.
            roc_auc = roc_auc if roc_auc is not None else None

    class_distribution = {
        str(c): int(np.sum(np.asarray(y_true) == c)) for c in classes
    }

    return {
        "task_type": "classification",
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "f1_weighted": f1_weighted,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": [str(c) for c in classes],
        "per_class": per_class,
        "class_distribution": class_distribution,
        "n_samples": int(len(y_true)),
    }


def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    residuals = y_true - y_pred

    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_true, y_pred))

    hist_counts, hist_edges = np.histogram(residuals, bins=20)

    return {
        "task_type": "regression",
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "residual_mean": float(np.mean(residuals)),
        "residual_std": float(np.std(residuals)),
        "residual_histogram": {
            "counts": hist_counts.tolist(),
            "bin_edges": hist_edges.tolist(),
        },
        "n_samples": int(len(y_true)),
    }


def evaluate_clustering(
    X: np.ndarray, labels: np.ndarray, true_labels: np.ndarray | None = None, inertia: float | None = None
) -> dict:
    """Real unsupervised clustering evaluation. Internal validation metrics
    (silhouette, Calinski-Harabasz, Davies-Bouldin) always computed — they
    judge cluster separation using only the feature space, no ground truth
    needed. External validation (Adjusted Rand Index, Normalized Mutual
    Info) computed only if the dataset happened to include a true label
    column to check clusters against."""
    labels = np.asarray(labels)
    unique_labels = np.unique(labels)
    n_clusters = int(len(unique_labels[unique_labels != -1]))  # -1 = noise, e.g. DBSCAN
    n_noise = int(np.sum(labels == -1))

    cluster_sizes = {
        str(lbl): int(np.sum(labels == lbl)) for lbl in unique_labels
    }

    result: dict = {
        "task_type": "clustering",
        "n_samples": int(len(labels)),
        "n_clusters_found": n_clusters,
        "n_noise_points": n_noise,
        "cluster_sizes": cluster_sizes,
        "inertia": float(inertia) if inertia is not None else None,
    }

    # Internal validation metrics require at least 2 clusters and fewer
    # clusters than samples to be mathematically defined.
    valid_mask = labels != -1
    n_valid_clusters = len(np.unique(labels[valid_mask])) if valid_mask.any() else 0
    if n_valid_clusters >= 2 and n_valid_clusters < valid_mask.sum():
        try:
            result["silhouette_score"] = float(silhouette_score(X[valid_mask], labels[valid_mask]))
        except Exception:
            result["silhouette_score"] = None
        try:
            result["calinski_harabasz_score"] = float(calinski_harabasz_score(X[valid_mask], labels[valid_mask]))
        except Exception:
            result["calinski_harabasz_score"] = None
        try:
            result["davies_bouldin_score"] = float(davies_bouldin_score(X[valid_mask], labels[valid_mask]))
        except Exception:
            result["davies_bouldin_score"] = None
    else:
        result["silhouette_score"] = None
        result["calinski_harabasz_score"] = None
        result["davies_bouldin_score"] = None

    if true_labels is not None:
        try:
            result["adjusted_rand_index"] = float(adjusted_rand_score(true_labels, labels))
            result["normalized_mutual_info"] = float(normalized_mutual_info_score(true_labels, labels))
        except Exception:
            result["adjusted_rand_index"] = None
            result["normalized_mutual_info"] = None
    else:
        result["adjusted_rand_index"] = None
        result["normalized_mutual_info"] = None

    return result
