"""Investigation orchestrator: runs the complete pipeline against real data
and a real loaded model, producing one consolidated result dict."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from . import adapters, calibration, dataset_profile, drift as drift_mod, error_analysis, evaluator, explanation, findings as findings_mod, slice_discovery


class InvestigationError(Exception):
    pass


def run_investigation(
    model_path: Path,
    dataset_path: Path,
    target_column: str,
    feature_columns: Optional[list[str]] = None,
    reference_dataset_path: Optional[Path] = None,
    progress_cb=None,
) -> dict:
    def report(pct, msg):
        if progress_cb:
            progress_cb(pct, msg)

    report(5, "Loading model")
    loaded = adapters.load_model(model_path)

    report(10, "Loading dataset")
    df = pd.read_csv(dataset_path)
    if target_column not in df.columns:
        raise InvestigationError(f"Target column '{target_column}' not found in dataset columns: {list(df.columns)}")

    report(20, "Profiling dataset")
    profile = dataset_profile.profile_dataset(df, target_column=target_column)

    report(25, "Validating model/dataset compatibility")
    compatible_features = adapters.validate_compatibility(loaded, list(df.columns), target_column)
    features = feature_columns if feature_columns else compatible_features
    features = [f for f in features if f in df.columns and f != target_column]
    if not features:
        raise InvestigationError("No usable feature columns after validation.")

    X = df[features].copy()
    y = df[target_column].values

    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            try:
                X[col] = pd.to_numeric(X[col])
            except Exception:
                pass

    report(25, "Running predictions")
    try:
        y_pred = adapters.predict(loaded, X)
    except Exception as e:
        raise InvestigationError(f"Model failed to generate predictions on this dataset: {e}") from e

    proba = None
    if loaded.info.task_type == "classification":
        try:
            proba = adapters.predict_proba(loaded, X)
        except Exception:
            proba = None

    classes = loaded.info.classes if loaded.info.classes else sorted(pd.unique(y).tolist())

    report(35, "Computing evaluation metrics")
    if loaded.info.task_type == "classification":
        evaluation = evaluator.evaluate_classification(y, y_pred, proba, classes)
    else:
        evaluation = evaluator.evaluate_regression(y, y_pred)

    report(45, "Analyzing errors")
    err_df = error_analysis.build_error_table(X, y, y_pred, loaded.info.task_type, proba, classes)
    error_examples = error_analysis.top_error_examples(err_df, loaded.info.task_type)
    error_conc = error_analysis.error_concentration(err_df, features)

    report(55, "Discovering problematic slices")
    slices = slice_discovery.discover_slices(err_df, features)

    report(65, "Analyzing confidence and calibration")
    confidence = (
        calibration.confidence_analysis(err_df)
        if loaded.info.task_type == "classification"
        else {"available": False, "reason": "Confidence/calibration analysis applies to classifiers only."}
    )

    report(72, "Computing feature importance")
    try:
        importance = explanation.global_feature_importance(loaded.estimator, X, y, features)
    except Exception as e:
        importance = {"available": False, "reason": str(e)}

    drift_results = None
    ood_results = None
    reference_evaluation = None
    if reference_dataset_path is not None:
        report(80, "Analyzing data drift")
        ref_df = pd.read_csv(reference_dataset_path)
        ref_features = [f for f in features if f in ref_df.columns]
        drift_results = drift_mod.analyze_drift(ref_df, df, ref_features)

        report(88, "Detecting out-of-distribution samples")
        try:
            ood_results = explanation.detect_ood_samples(ref_df[ref_features], X[ref_features], ref_features)
        except Exception as e:
            ood_results = {"available": False, "reason": str(e)}

        if target_column in ref_df.columns:
            report(92, "Evaluating reference-period performance")
            ref_X = ref_df[features].copy()
            ref_y = ref_df[target_column].values
            for col in ref_X.columns:
                if not pd.api.types.is_numeric_dtype(ref_X[col]):
                    try:
                        ref_X[col] = pd.to_numeric(ref_X[col])
                    except Exception:
                        pass
            try:
                ref_pred = adapters.predict(loaded, ref_X)
                if loaded.info.task_type == "classification":
                    ref_proba = None
                    try:
                        ref_proba = adapters.predict_proba(loaded, ref_X)
                    except Exception:
                        pass
                    reference_evaluation = evaluator.evaluate_classification(ref_y, ref_pred, ref_proba, classes)
                else:
                    reference_evaluation = evaluator.evaluate_regression(ref_y, ref_pred)
            except Exception:
                reference_evaluation = None

    report(95, "Generating findings")
    generated_findings = findings_mod.generate_findings(
        evaluation, error_conc, slices, confidence, drift_results, reference_evaluation
    )

    report(100, "Done")
    return {
        "model_info": {
            "task_type": loaded.info.task_type,
            "estimator_class": loaded.info.estimator_class,
            "supports_proba": loaded.info.supports_proba,
            "classes": [str(c) for c in classes] if loaded.info.task_type == "classification" else None,
        },
        "dataset_info": {
            "n_samples": int(len(df)),
            "n_features_used": len(features),
            "feature_columns": features,
            "target_column": target_column,
        },
        "dataset_profile": profile,
        "evaluation": evaluation,
        "reference_evaluation": reference_evaluation,
        "error_examples": error_examples,
        "error_concentration": error_conc,
        "slices": slices,
        "confidence": confidence,
        "feature_importance": importance,
        "drift": drift_results,
        "ood": ood_results,
        "findings": generated_findings,
    }
