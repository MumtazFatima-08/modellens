import numpy as np
from app.ml_engine import evaluator


def test_classification_perfect():
    y_true = np.array([0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1, 1])
    result = evaluator.evaluate_classification(y_true, y_pred, None, [0, 1])
    assert result["accuracy"] == 1.0
    assert result["f1_macro"] == 1.0
    assert result["confusion_matrix"] == [[2, 0], [0, 3]]


def test_classification_with_errors():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 0])
    result = evaluator.evaluate_classification(y_true, y_pred, None, [0, 1])
    assert result["accuracy"] == 0.5
    assert result["confusion_matrix"][0][0] == 1  # true 0 pred 0
    assert result["n_samples"] == 4


def test_classification_roc_auc_binary():
    y_true = np.array([0, 0, 1, 1])
    proba = np.array([[0.9, 0.1], [0.6, 0.4], [0.3, 0.7], [0.1, 0.9]])
    y_pred = np.array([0, 0, 1, 1])
    result = evaluator.evaluate_classification(y_true, y_pred, proba, [0, 1])
    assert result["roc_auc"] == 1.0


def test_regression_metrics():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.1, 1.9, 3.2, 3.8])
    result = evaluator.evaluate_regression(y_true, y_pred)
    assert result["mae"] > 0
    assert result["rmse"] >= result["mae"]  # RMSE >= MAE always holds
    assert -1.0 <= result["r2"] <= 1.0
