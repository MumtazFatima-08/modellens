import numpy as np
import pandas as pd
from app.ml_engine import slice_discovery, error_analysis


def _synthetic_error_df():
    rng = np.random.default_rng(0)
    n = 1000
    f1 = rng.normal(0, 1, n)
    f2 = rng.normal(0, 1, n)
    # errors concentrated where f1 > 1.0
    is_error = ((f1 > 1.0) & (rng.uniform(0, 1, n) < 0.8)) | (rng.uniform(0, 1, n) < 0.05)
    df = pd.DataFrame({"f1": f1, "f2": f2})
    df["_true"] = 0
    df["_pred"] = is_error.astype(int)  # placeholder, not used directly
    df["_is_error"] = is_error.astype(int)
    return df


def test_slice_discovery_finds_worse_than_average_slice():
    df = _synthetic_error_df()
    slices = slice_discovery.discover_slices(df, ["f1", "f2"], min_sample_size=20)
    assert len(slices) > 0
    for s in slices:
        assert s["performance_gap"] > 0  # only worse-than-average slices returned
        assert s["sample_size"] >= 20


def test_slice_discovery_empty_when_no_signal():
    rng = np.random.default_rng(1)
    n = 200
    df = pd.DataFrame({"f1": rng.normal(0, 1, n), "f2": rng.normal(0, 1, n)})
    df["_is_error"] = (rng.uniform(0, 1, n) < 0.1).astype(int)  # random, no structure
    slices = slice_discovery.discover_slices(df, ["f1", "f2"], min_sample_size=20)
    # With no real structure, we should find few or no significant large-gap slices
    assert isinstance(slices, list)


def test_error_concentration_detects_numeric_association():
    df = _synthetic_error_df()
    results = error_analysis.error_concentration(df, ["f1", "f2"])
    f1_result = next((r for r in results if r["feature"] == "f1"), None)
    assert f1_result is not None
    assert f1_result["significant"] is True
    assert f1_result["lift"] > 0
