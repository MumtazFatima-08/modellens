"""
Generates real synthetic datasets for each ModelLens sample case.
Run: python scripts/generate_sample_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "sample_data"
OUT.mkdir(exist_ok=True)

RNG = 42


def _to_df(X, y, n_numeric, n_categorical_from_numeric=0):
    cols = [f"feature_{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    # turn a couple of columns into interpretable categorical-ish bins for realism
    if n_categorical_from_numeric:
        for i in range(n_categorical_from_numeric):
            col = cols[-(i + 1)]
            df[col] = pd.qcut(df[col], q=4, labels=["low", "medium", "high", "very_high"]).astype(str)
    df["target"] = y
    return df


def gen_healthy_classification():
    X, y = make_classification(
        n_samples=2000, n_features=10, n_informative=7, n_redundant=1,
        n_classes=2, class_sep=1.8, flip_y=0.02, random_state=RNG,
    )
    df = _to_df(X, y, 10, n_categorical_from_numeric=1)
    df.to_csv(OUT / "healthy_classification_current.csv", index=False)


def gen_weak_classification():
    X, y = make_classification(
        n_samples=2000, n_features=10, n_informative=3, n_redundant=4,
        n_classes=2, class_sep=0.5, flip_y=0.25, random_state=RNG,
    )
    df = _to_df(X, y, 10, n_categorical_from_numeric=1)
    df.to_csv(OUT / "weak_classification_current.csv", index=False)


def gen_imbalanced_classification():
    X, y = make_classification(
        n_samples=3000, n_features=8, n_informative=5, n_redundant=1,
        n_classes=2, weights=[0.95, 0.05], flip_y=0.02, class_sep=1.2, random_state=RNG,
    )
    df = _to_df(X, y, 8, n_categorical_from_numeric=1)
    df.to_csv(OUT / "imbalanced_classification_current.csv", index=False)


def gen_distribution_shift():
    # Reference: "training-period" distribution
    X_ref, y_ref = make_classification(
        n_samples=1500, n_features=8, n_informative=5, n_redundant=1,
        n_classes=2, class_sep=1.5, flip_y=0.03, random_state=RNG,
    )
    df_ref = _to_df(X_ref, y_ref, 8, n_categorical_from_numeric=1)
    df_ref.to_csv(OUT / "distribution_shift_reference.csv", index=False)

    # Current: same generative process but with a real, deliberate shift in
    # several numeric features (mean shift + variance increase) applied
    # AFTER label assignment logic diverges slightly (simulate real drift).
    rng = np.random.default_rng(RNG + 1)
    X_cur, y_cur = make_classification(
        n_samples=1500, n_features=8, n_informative=5, n_redundant=1,
        n_classes=2, class_sep=1.5, flip_y=0.10, random_state=RNG + 1,
    )
    # inject real distribution shift on a subset of numeric features
    X_cur[:, 0] = X_cur[:, 0] * 1.8 + 2.5      # mean + variance shift
    X_cur[:, 2] = X_cur[:, 2] + 3.0            # mean shift
    X_cur[:, 4] = X_cur[:, 4] * 2.5            # variance shift
    df_cur = _to_df(X_cur, y_cur, 8, n_categorical_from_numeric=1)
    df_cur.to_csv(OUT / "distribution_shift_current.csv", index=False)


def gen_error_prone():
    # Errors will genuinely cluster: model trained WITHOUT feature_0/feature_1
    # interaction region, dataset includes a subgroup driven by an XOR-like
    # interaction that a simple model structurally cannot capture well.
    rng = np.random.default_rng(RNG)
    n = 2500
    f0 = rng.normal(0, 1, n)
    f1 = rng.normal(0, 1, n)
    f2 = rng.normal(0, 1, n)
    f3 = rng.normal(0, 1, n)
    others = rng.normal(0, 1, (n, 5))

    base_logit = 1.5 * f2 - 1.2 * f3
    xor_region = (f0 > 0.5) & (f1 > 0.5)
    # In the xor_region, flip the true relationship — this is the region a
    # smooth/simple model trained on the whole data will systematically miss.
    logit = np.where(xor_region, -(1.5 * f0 + 1.5 * f1) + base_logit * 0.2, base_logit + 0.6 * f0)
    prob = 1 / (1 + np.exp(-logit))
    y = (rng.uniform(0, 1, n) < prob).astype(int)

    X = np.column_stack([f0, f1, f2, f3, others])
    df = _to_df(X, y, 9, n_categorical_from_numeric=1)
    df.to_csv(OUT / "error_prone_current.csv", index=False)


def gen_text_classification():
    """Real free-text data (product review style), no numeric features at
    all — proves ModelLens investigates text classifiers (sklearn Pipeline
    with a built-in vectorizer, e.g. TfidfVectorizer + LogisticRegression)
    through the exact same upload flow as tabular models."""
    rng = np.random.default_rng(RNG)

    positive_phrases = [
        "absolutely love this product", "works perfectly every time", "exceeded my expectations",
        "great value for the price", "highly recommend to everyone", "fantastic build quality",
        "customer service was wonderful", "arrived quickly and well packaged", "exactly what i needed",
        "best purchase i made this year", "so happy with this", "five stars would buy again",
        "impressive performance and reliable", "easy to use and well designed", "worth every penny",
    ]
    negative_phrases = [
        "completely disappointed with this", "stopped working after two days", "waste of money honestly",
        "terrible customer service experience", "arrived broken and late", "not as described at all",
        "poor build quality overall", "would not recommend to anyone", "regret buying this product",
        "cheaply made and flimsy", "worst purchase i made this year", "one star this is awful",
        "unreliable and frustrating to use", "difficult to set up and confusing", "not worth the price",
    ]
    filler = ["", " for the price point", " compared to competitors", " after a week of use",
              " based on my experience", " as a first time buyer", ""]

    n = 1600
    texts, labels = [], []
    for _ in range(n):
        is_positive = rng.uniform() < 0.5
        bank = positive_phrases if is_positive else negative_phrases
        phrase = str(rng.choice(bank))
        f = str(rng.choice(filler))
        # occasional label noise so the model isn't trivially perfect
        flip = rng.uniform() < 0.06
        texts.append(phrase + f)
        labels.append(int(not is_positive) if flip else int(is_positive))

    df = pd.DataFrame({"review_text": texts, "target": labels})
    df.to_csv(OUT / "text_classification_current.csv", index=False)


def gen_regression():
    X, y = make_regression(
        n_samples=2000, n_features=8, n_informative=5, noise=12.0, random_state=RNG,
    )
    df = _to_df(X, y, 8, n_categorical_from_numeric=1)
    df.to_csv(OUT / "regression_current.csv", index=False)


if __name__ == "__main__":
    gen_healthy_classification()
    gen_weak_classification()
    gen_imbalanced_classification()
    gen_distribution_shift()
    gen_error_prone()
    gen_text_classification()
    gen_regression()
    print(f"Sample datasets written to {OUT}")
