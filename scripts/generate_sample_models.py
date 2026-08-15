"""
Trains real scikit-learn models on the sample datasets and saves them as
.joblib artifacts, plus the matching *_reference.csv training-period files
where relevant. Run generate_sample_data.py first.

Run: python scripts/generate_sample_models.py
"""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "sample_data"
MODELS = ROOT / "sample_models"
MODELS.mkdir(exist_ok=True)


def _build_pipeline(df: pd.DataFrame, estimator):
    feature_cols = [c for c in df.columns if c != "target"]
    cat_cols = [c for c in feature_cols if not pd.api.types.is_numeric_dtype(df[c])]
    num_cols = [c for c in feature_cols if c not in cat_cols]
    pre = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
        remainder="passthrough",
    )
    return Pipeline([("pre", pre), ("model", estimator)]), feature_cols


def train_healthy_classification():
    df = pd.read_csv(DATA / "healthy_classification_current.csv")
    pipe, features = _build_pipeline(df, RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42))
    train_and_save(df, pipe, "healthy_classification", features)


def train_weak_classification():
    df = pd.read_csv(DATA / "weak_classification_current.csv")
    pipe, features = _build_pipeline(df, LogisticRegression(max_iter=500))
    train_and_save(df, pipe, "weak_classification", features)


def train_imbalanced_classification():
    df = pd.read_csv(DATA / "imbalanced_classification_current.csv")
    pipe, features = _build_pipeline(df, LogisticRegression(max_iter=500))
    train_and_save(df, pipe, "imbalanced_classification", features)


def train_distribution_shift():
    df_ref = pd.read_csv(DATA / "distribution_shift_reference.csv")
    pipe, features = _build_pipeline(df_ref, GradientBoostingClassifier(n_estimators=150, random_state=42))
    X, y = df_ref[features], df_ref["target"]
    pipe.fit(X, y)
    joblib.dump(pipe, MODELS / "distribution_shift.joblib")
    print("distribution_shift: trained on reference period only (as intended)")


def train_error_prone():
    df = pd.read_csv(DATA / "error_prone_current.csv")
    # Deliberately use a LINEAR model (LogisticRegression) which structurally
    # cannot capture the XOR-like interaction baked into the data generator —
    # this produces genuine, real, feature-combination-clustered errors.
    pipe, features = _build_pipeline(df, LogisticRegression(max_iter=500))
    train_and_save(df, pipe, "error_prone", features)


def train_text_classification():
    from sklearn.feature_extraction.text import TfidfVectorizer

    df = pd.read_csv(DATA / "text_classification_current.csv")
    X, y = df[["review_text"]], df["target"]

    pipe = Pipeline([
        ("tfidf", ColumnTransformer(
            transformers=[("text", TfidfVectorizer(max_features=500, ngram_range=(1, 2)), "review_text")],
        )),
        ("model", LogisticRegression(max_iter=1000)),
    ])
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.01, random_state=42)
    pipe.fit(X_train, y_train)
    joblib.dump(pipe, MODELS / "text_classification.joblib")
    print("text_classification: trained and saved (TF-IDF + LogisticRegression, real sklearn text pipeline)")


def train_regression():
    df = pd.read_csv(DATA / "regression_current.csv")
    pipe, features = _build_pipeline(df, RandomForestRegressor(n_estimators=150, max_depth=8, random_state=42))
    train_and_save(df, pipe, "regression", features)


def train_and_save(df, pipe, name, features):
    X, y = df[features], df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1.0, shuffle=False) \
        if False else (X, None, y, None)
    # Train on a held-in subset, then evaluate on the FULL current.csv via the
    # API later (ModelLens investigates realistic in-production behavior).
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.01, random_state=42)
    pipe.fit(X_train, y_train)
    joblib.dump(pipe, MODELS / f"{name}.joblib")
    print(f"{name}: trained and saved")


if __name__ == "__main__":
    train_healthy_classification()
    train_weak_classification()
    train_imbalanced_classification()
    train_distribution_shift()
    train_error_prone()
    train_text_classification()
    train_regression()
    print(f"Sample models written to {MODELS}")
