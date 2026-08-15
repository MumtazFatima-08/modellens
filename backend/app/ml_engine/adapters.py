"""
Model adapter layer.

Wraps a loaded, arbitrary scikit-learn-compatible estimator behind a small,
stable interface so the rest of the engine never has to special-case model
types. Supports anything exposing the standard sklearn estimator API
(predict, optionally predict_proba). XGBoost/LightGBM sklearn wrappers are
supported automatically because they implement this same interface.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
from sklearn.base import ClusterMixin
from sklearn.pipeline import Pipeline


class UnsupportedModelError(Exception):
    pass


class ModelDatasetMismatchError(Exception):
    pass


@dataclass
class ModelInfo:
    task_type: str  # "classification" | "regression"
    n_features_expected: Optional[int]
    feature_names_expected: Optional[list[str]]
    classes: Optional[list[Any]]
    estimator_class: str
    supports_proba: bool
    supports_native_importance: bool


@dataclass
class LoadedModel:
    estimator: Any
    info: ModelInfo
    path: Path


def load_model(path: Path) -> LoadedModel:
    """Load a .pkl/.joblib file and introspect it. Raises UnsupportedModelError
    if the object doesn't look like a usable estimator."""
    suffix = path.suffix.lower()
    try:
        if suffix == ".joblib":
            estimator = joblib.load(path)
        elif suffix == ".pkl":
            try:
                estimator = joblib.load(path)
            except Exception:
                with open(path, "rb") as f:
                    estimator = pickle.load(f)
        else:
            raise UnsupportedModelError(
                f"Unsupported model file extension '{suffix}'. Expected .pkl or .joblib."
            )
    except UnsupportedModelError:
        raise
    except Exception as e:
        raise UnsupportedModelError(f"Could not deserialize model file: {e}") from e

    if not hasattr(estimator, "predict"):
        raise UnsupportedModelError(
            "Loaded object does not expose a .predict() method. "
            "ModelLens requires a scikit-learn-compatible estimator "
            "(or a Pipeline ending in one)."
        )

    supports_proba = hasattr(estimator, "predict_proba")
    is_classifier = supports_proba or hasattr(estimator, "classes_")

    # Clustering detection: recognize common unsupervised clusterers by
    # their sklearn attributes (cluster_centers_/labels_) rather than a
    # hardcoded name list, so custom subclasses are still picked up.
    final_estimator = estimator
    if hasattr(estimator, "named_steps"):  # unwrap a Pipeline to inspect the final step
        try:
            final_estimator = list(estimator.named_steps.values())[-1]
        except Exception:
            final_estimator = estimator

    is_clusterer = (
        hasattr(final_estimator, "cluster_centers_")
        or hasattr(final_estimator, "labels_")
        or type(final_estimator).__name__
        in {"KMeans", "MiniBatchKMeans", "DBSCAN", "AgglomerativeClustering", "GaussianMixture", "SpectralClustering", "Birch", "MeanShift", "OPTICS"}
    )

    if is_clusterer and not is_classifier:
        task_type = "clustering"
    elif is_classifier:
        task_type = "classification"
    else:
        task_type = "regression"

    n_features_expected = getattr(estimator, "n_features_in_", None)
    feature_names_expected = None
    fn = getattr(estimator, "feature_names_in_", None)
    if fn is not None:
        feature_names_expected = list(fn)

    classes = None
    if is_classifier and hasattr(estimator, "classes_"):
        classes = list(estimator.classes_)

    supports_native_importance = hasattr(estimator, "feature_importances_") or hasattr(
        estimator, "coef_"
    )

    info = ModelInfo(
        task_type=task_type,
        n_features_expected=int(n_features_expected) if n_features_expected is not None else None,
        feature_names_expected=feature_names_expected,
        classes=classes,
        estimator_class=type(estimator).__name__,
        supports_proba=supports_proba,
        supports_native_importance=supports_native_importance,
    )
    return LoadedModel(estimator=estimator, info=info, path=path)


def validate_compatibility(loaded: LoadedModel, dataset_columns: list[str], target_column: str) -> list[str]:
    """Returns a list of candidate feature columns usable with this model,
    raising ModelDatasetMismatchError with a human-readable message if the
    dataset is fundamentally incompatible."""
    available = [c for c in dataset_columns if c != target_column]

    if loaded.info.feature_names_expected:
        expected = loaded.info.feature_names_expected
        compatible = [c for c in expected if c in available]
        if len(compatible) == 0:
            raise ModelDatasetMismatchError(
                f"Model expects features {expected} but none of these columns "
                f"were found in the uploaded dataset (available columns: {available})."
            )
        if len(compatible) < len(expected):
            missing = [c for c in expected if c not in available]
            raise ModelDatasetMismatchError(
                f"Model expects {len(expected)} named features but dataset is "
                f"missing {len(missing)} of them: {missing}."
            )
        return compatible

    if loaded.info.n_features_expected is not None:
        if len(available) < loaded.info.n_features_expected:
            raise ModelDatasetMismatchError(
                f"Model expects {loaded.info.n_features_expected} features but dataset "
                f"contains only {len(available)} candidate columns (excluding target)."
            )
        if len(available) > loaded.info.n_features_expected:
            raise ModelDatasetMismatchError(
                f"Model expects {loaded.info.n_features_expected} features but dataset "
                f"contains {len(available)} candidate columns. Select exactly "
                f"{loaded.info.n_features_expected} feature columns explicitly."
            )
        return available

    return available


def predict(loaded: LoadedModel, X: np.ndarray) -> np.ndarray:
    if not hasattr(loaded.estimator, "predict"):
        raise UnsupportedModelError(
            f"This {loaded.info.estimator_class} does not support predicting cluster "
            "assignments for new data (only DBSCAN/AgglomerativeClustering-style "
            "estimators have this limitation — they only label the data they were "
            "originally fit on). Use KMeans, MiniBatchKMeans, or GaussianMixture "
            "for investigations on a separate evaluation dataset."
        )
    return loaded.estimator.predict(X)


def predict_proba(loaded: LoadedModel, X: np.ndarray) -> Optional[np.ndarray]:
    if not loaded.info.supports_proba:
        return None
    return loaded.estimator.predict_proba(X)
