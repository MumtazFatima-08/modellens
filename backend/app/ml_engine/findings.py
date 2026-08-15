"""Findings generator: converts already-computed statistical evidence into
structured findings (severity, evidence, interpretation, recommendation).
No numbers are invented here — every field is templated from values passed
in from the evaluator/error_analysis/slice_discovery/drift/calibration
modules. Language is deliberately hedged (associated with / potential /
observed) per ModelLens's scientific-honesty requirement."""
from __future__ import annotations


def generate_findings(
    evaluation: dict,
    error_concentration: list[dict],
    slices: list[dict],
    confidence: dict,
    drift: list[dict] | None,
    reference_evaluation: dict | None,
) -> list[dict]:
    findings = []

    # --- High-confidence errors ---
    if confidence.get("available") and confidence["n_total_errors"] > 0:
        pct = confidence["pct_errors_high_confidence"]
        if pct >= 0.15:
            severity = "critical" if pct >= 0.30 else "warning"
            findings.append(
                {
                    "severity": severity,
                    "title": "High-confidence incorrect predictions detected",
                    "evidence": {
                        "pct_errors_high_confidence": round(pct * 100, 1),
                        "n_high_confidence_errors": confidence["n_high_confidence_errors"],
                        "n_total_errors": confidence["n_total_errors"],
                        "median_confidence_of_these_errors": confidence.get(
                            "median_confidence_of_high_conf_errors"
                        ),
                        "confidence_threshold": confidence["high_confidence_threshold"],
                    },
                    "interpretation": (
                        f"{round(pct * 100, 1)}% of incorrect predictions had confidence at or above "
                        f"{int(confidence['high_confidence_threshold']*100)}%. This pattern is associated with "
                        "poor calibration or systematic overconfidence in specific regions of the feature space, "
                        "rather than with borderline/ambiguous cases the model itself is unsure about."
                    ),
                    "recommendation": (
                        "Investigate the calibration tab and consider recalibrating (Platt scaling or isotonic "
                        "regression) before trusting confidence scores for downstream decisions."
                    ),
                }
            )

    # --- Calibration error ---
    if confidence.get("available") and confidence.get("expected_calibration_error") is not None:
        ece = confidence["expected_calibration_error"]
        if ece >= 0.05:
            severity = "critical" if ece >= 0.15 else "warning"
            findings.append(
                {
                    "severity": severity,
                    "title": "Model confidence appears poorly calibrated",
                    "evidence": {"expected_calibration_error": round(ece, 4)},
                    "interpretation": (
                        f"Expected Calibration Error is {round(ece, 4)}. Predicted confidence does not closely "
                        "track observed accuracy across confidence bins, meaning stated probabilities may not be "
                        "trustworthy at face value."
                    ),
                    "recommendation": "Review the reliability diagram; consider a calibration step for probability-dependent use cases.",
                }
            )

    # --- Error concentration ---
    sig_concentration = [e for e in error_concentration if e.get("significant")]
    for e in sig_concentration[:3]:
        gap_pct = round(e["lift"] * 100, 1)
        findings.append(
            {
                "severity": "warning" if abs(e["lift"]) < 0.25 else "critical",
                "title": f"Errors are associated with '{e['feature']}'",
                "evidence": {
                    "condition": e["condition"],
                    "group_error_rate": round(e["group_error_rate"] * 100, 1),
                    "overall_error_rate": round(e["overall_error_rate"] * 100, 1),
                    "group_size": e["group_size"],
                    "p_value": e["p_value"],
                },
                "interpretation": (
                    f"Samples where {e['condition']} show a {round(e['group_error_rate']*100,1)}% error rate "
                    f"versus {round(e['overall_error_rate']*100,1)}% overall (statistically significant, "
                    f"p={round(e['p_value'], 4)}). This is an observed association, not a proven cause of failure."
                ),
                "recommendation": f"Inspect examples in this region and consider targeted data collection or feature engineering around '{e['feature']}'.",
            }
        )

    # --- Problematic slices ---
    for s in [sl for sl in slices if sl.get("significant")][:3]:
        findings.append(
            {
                "severity": "critical" if s["performance_gap"] > 0.25 else "warning",
                "title": "Candidate problematic slice discovered",
                "evidence": {
                    "rule": s["rule"],
                    "sample_size": s["sample_size"],
                    "group_error_rate": round(s["group_error_rate"] * 100, 1),
                    "overall_error_rate": round(s["overall_error_rate"] * 100, 1),
                    "performance_gap": round(s["performance_gap"] * 100, 1),
                    "p_value": s["p_value"],
                },
                "interpretation": (
                    f"The subgroup defined by [{s['rule']}] ({s['sample_size']} samples) shows a "
                    f"{round(s['performance_gap']*100,1)} percentage-point higher error rate than the overall "
                    "model. Flagged as a candidate problematic slice pending further investigation."
                ),
                "recommendation": "Validate with a larger sample if possible; consider slice-specific model improvements or guardrails.",
            }
        )

    # --- Drift ---
    if drift:
        severe = [d for d in drift if d["status"] == "severe_drift"]
        moderate = [d for d in drift if d["status"] == "drift"]
        if severe:
            findings.append(
                {
                    "severity": "critical",
                    "title": "Severe distribution drift detected",
                    "evidence": {"features": [d["feature"] for d in severe], "psi_scores": {d["feature"]: round(d["psi"], 3) for d in severe}},
                    "interpretation": (
                        f"{len(severe)} feature(s) show severe distribution shift (PSI ≥ 0.25) between the "
                        "reference and current data: " + ", ".join(d["feature"] for d in severe) + ". "
                        "Drift does not automatically mean model failure — check whether it correlates with "
                        "the performance-degradation and slice findings above."
                    ),
                    "recommendation": "Cross-reference drifted features against discovered problematic slices; consider retraining if drift persists.",
                }
            )
        elif moderate:
            findings.append(
                {
                    "severity": "info",
                    "title": "Moderate distribution drift observed",
                    "evidence": {"features": [d["feature"] for d in moderate]},
                    "interpretation": f"{len(moderate)} feature(s) show moderate drift (0.1 ≤ PSI < 0.25). Worth monitoring.",
                    "recommendation": "Continue monitoring; no immediate action required unless paired with performance degradation.",
                }
            )

    # --- Performance degradation vs reference ---
    if reference_evaluation and evaluation.get("task_type") == reference_evaluation.get("task_type"):
        if evaluation["task_type"] == "classification":
            cur_f1 = evaluation.get("f1_macro")
            ref_f1 = reference_evaluation.get("f1_macro")
            if cur_f1 is not None and ref_f1 is not None and ref_f1 > 0:
                change_pct = (cur_f1 - ref_f1) / ref_f1 * 100
                if change_pct <= -5:
                    findings.append(
                        {
                            "severity": "critical" if change_pct <= -15 else "warning",
                            "title": "Performance degradation vs. reference",
                            "evidence": {
                                "reference_f1_macro": round(ref_f1, 4),
                                "current_f1_macro": round(cur_f1, 4),
                                "change_pct": round(change_pct, 1),
                            },
                            "interpretation": f"Macro F1 changed from {round(ref_f1,3)} to {round(cur_f1,3)} ({round(change_pct,1)}%), an observed degradation relative to the reference dataset's performance.",
                            "recommendation": "Investigate whether this correlates with the drift and slice findings above before deciding to retrain.",
                        }
                    )
        else:
            cur_r2 = evaluation.get("r2")
            ref_r2 = reference_evaluation.get("r2")
            if cur_r2 is not None and ref_r2 is not None:
                delta = cur_r2 - ref_r2
                if delta <= -0.05:
                    findings.append(
                        {
                            "severity": "critical" if delta <= -0.15 else "warning",
                            "title": "Performance degradation vs. reference (R²)",
                            "evidence": {"reference_r2": round(ref_r2, 4), "current_r2": round(cur_r2, 4), "delta": round(delta, 4)},
                            "interpretation": f"R² dropped from {round(ref_r2,3)} to {round(cur_r2,3)} relative to the reference dataset.",
                            "recommendation": "Investigate whether this correlates with drift or specific slices before retraining.",
                        }
                    )

    # If nothing notable found, say so honestly.
    if not findings:
        findings.append(
            {
                "severity": "info",
                "title": "No significant issues detected",
                "evidence": {},
                "interpretation": "Within the statistical thresholds used by this investigation, no high-confidence error clusters, significant slices, severe drift, or performance degradation were detected.",
                "recommendation": "Continue routine monitoring. Absence of detected issues is not a guarantee of reliability outside the tested distribution.",
            }
        )

    severity_order = {"critical": 0, "warning": 1, "info": 2}
    findings.sort(key=lambda f: severity_order.get(f["severity"], 3))
    return findings
