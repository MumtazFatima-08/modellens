import pandas as pd
from app.ml_engine import dataset_profile


def test_profile_basic_columns():
    df = pd.DataFrame({
        "num": [1, 2, 3, 4, None],
        "cat": ["a", "a", "b", "b", "c"],
        "target": [0, 1, 0, 1, 0],
    })
    profile = dataset_profile.profile_dataset(df, target_column="target")
    assert profile["n_rows"] == 5
    assert profile["n_columns"] == 3

    num_col = next(c for c in profile["columns"] if c["name"] == "num")
    assert num_col["dtype"] == "numeric"
    assert num_col["n_missing"] == 1
    assert num_col["min"] == 1.0
    assert num_col["max"] == 4.0

    cat_col = next(c for c in profile["columns"] if c["name"] == "cat")
    assert cat_col["dtype"] == "categorical"
    assert cat_col["n_unique"] == 3
    assert cat_col["top_values"][0]["value"] in ("a", "b")


def test_profile_detects_imbalance():
    df = pd.DataFrame({"x": range(100), "target": [0] * 95 + [1] * 5})
    profile = dataset_profile.profile_dataset(df, target_column="target")
    assert profile["target_summary"]["type"] == "categorical"
    assert profile["target_summary"]["is_imbalanced"] is True
    assert profile["target_summary"]["class_counts"]["1"] == 5


def test_profile_detects_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [1, 1, 2]})
    profile = dataset_profile.profile_dataset(df)
    assert profile["duplicate_rows"] == 1


def test_profile_continuous_target():
    df = pd.DataFrame({"x": range(50), "target": [i * 1.5 for i in range(50)]})
    profile = dataset_profile.profile_dataset(df, target_column="target")
    assert profile["target_summary"]["type"] == "continuous"
