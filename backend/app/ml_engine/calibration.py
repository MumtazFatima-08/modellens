"""Confidence vs correctness, calibration analysis, high-confidence-error
detection. All computed from real predicted probabilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve


def confidence_analysis(err_df: pd.DataFrame, n_bins: int = 10) -> dict:
    if "_confidence" not in err_df.columns or err_df["_confidence"].isna().all():
        return {"available": False}

    conf = err_df["_confidence"].astype(float)
    is_error = err_df["_is_error"].astype(int)
    is_correct = 1 - is_error

    correct_conf = conf[is_correct == 1]
    incorrect_conf = conf[is_correct == 0]

    high_conf_threshold = 0.9
    high_conf_mask = conf >= high_conf_threshold
    high_conf_errors = err_df[high_conf_mask & (is_error == 1)]
    n_high_conf_errors = int(len(high_conf_errors))
    n_total_errors = int(is_error.sum())

    reliability_true, reliability_pred = calibration_curve(
        is_correct, conf, n_bins=n_bins, strategy="uniform"
    )

    # Expected Calibration Error
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_idx = np.digitize(conf, bin_edges[1:-1])
    ece = 0.0
    bin_details = []
    for b in range(n_bins):
        mask = bin_idx == b
        n_in_bin = int(mask.sum())
        if n_in_bin == 0:
            continue
        avg_conf = float(conf[mask].mean())
        avg_acc = float(is_correct[mask].mean())
        ece += (n_in_bin / len(conf)) * abs(avg_conf - avg_acc)
        bin_details.append(
            {
                "bin_range": [float(bin_edges[b]), float(bin_edges[b + 1])],
                "avg_confidence": avg_conf,
                "avg_accuracy": avg_acc,
                "n_samples": n_in_bin,
            }
        )

    hist_counts, hist_edges = np.histogram(conf, bins=20, range=(0, 1))
    correct_hist, _ = np.histogram(correct_conf, bins=hist_edges) if len(correct_conf) else (np.zeros(20), hist_edges)
    incorrect_hist, _ = np.histogram(incorrect_conf, bins=hist_edges) if len(incorrect_conf) else (np.zeros(20), hist_edges)

    return {
        "available": True,
        "mean_confidence_correct": float(correct_conf.mean()) if len(correct_conf) else None,
        "mean_confidence_incorrect": float(incorrect_conf.mean()) if len(incorrect_conf) else None,
        "high_confidence_threshold": high_conf_threshold,
        "n_high_confidence_errors": n_high_conf_errors,
        "n_total_errors": n_total_errors,
        "pct_errors_high_confidence": float(n_high_conf_errors / n_total_errors) if n_total_errors > 0 else 0.0,
        "median_confidence_of_high_conf_errors": float(high_conf_errors["_confidence"].median())
        if n_high_conf_errors > 0
        else None,
        "expected_calibration_error": float(ece),
        "reliability_curve": {
            "predicted_confidence": reliability_pred.tolist(),
            "observed_accuracy": reliability_true.tolist(),
        },
        "calibration_bins": bin_details,
        "confidence_histogram": {
            "bin_edges": hist_edges.tolist(),
            "correct_counts": correct_hist.tolist() if hasattr(correct_hist, "tolist") else list(correct_hist),
            "incorrect_counts": incorrect_hist.tolist() if hasattr(incorrect_hist, "tolist") else list(incorrect_hist),
        },
    }
