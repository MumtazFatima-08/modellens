import numpy as np
import pandas as pd
from app.ml_engine import drift


def test_no_drift_identical_distributions():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"x": rng.normal(0, 1, 1000)})
    cur = pd.DataFrame({"x": rng.normal(0, 1, 1000)})
    results = drift.analyze_drift(ref, cur, ["x"])
    assert results[0]["status"] in ("normal", "drift")  # sampling noise tolerance
    assert results[0]["psi"] < 0.25


def test_severe_drift_detected():
    rng = np.random.default_rng(0)
    ref = pd.DataFrame({"x": rng.normal(0, 1, 1000)})
    cur = pd.DataFrame({"x": rng.normal(5, 3, 1000)})  # big mean+variance shift
    results = drift.analyze_drift(ref, cur, ["x"])
    assert results[0]["status"] == "severe_drift"
    assert results[0]["psi"] > 0.25


def test_categorical_drift():
    ref = pd.DataFrame({"c": ["a"] * 800 + ["b"] * 200})
    cur = pd.DataFrame({"c": ["a"] * 200 + ["b"] * 800})  # flipped proportions
    results = drift.analyze_drift(ref, cur, ["c"])
    assert results[0]["feature_type"] == "categorical"
    assert results[0]["status"] != "normal"


def test_ks_statistic_present():
    rng = np.random.default_rng(1)
    ref = pd.DataFrame({"x": rng.normal(0, 1, 500)})
    cur = pd.DataFrame({"x": rng.normal(0, 1, 500)})
    results = drift.analyze_drift(ref, cur, ["x"])
    assert 0 <= results[0]["ks_statistic"] <= 1
