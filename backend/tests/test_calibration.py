import numpy as np
import pandas as pd
from app.ml_engine import calibration


def test_confidence_analysis_well_calibrated():
    rng = np.random.default_rng(0)
    n = 2000
    conf = rng.uniform(0.5, 1.0, n)
    is_correct = (rng.uniform(0, 1, n) < conf).astype(int)
    df = pd.DataFrame({"_confidence": conf, "_is_error": 1 - is_correct})
    result = calibration.confidence_analysis(df)
    assert result["available"] is True
    assert 0 <= result["expected_calibration_error"] <= 1
    assert result["n_total_errors"] == int((1 - is_correct).sum())


def test_confidence_analysis_unavailable_without_proba():
    df = pd.DataFrame({"_is_error": [0, 1, 0, 1]})
    result = calibration.confidence_analysis(df)
    assert result["available"] is False


def test_high_confidence_errors_flagged():
    # deliberately overconfident-and-wrong model
    n = 500
    conf = np.full(n, 0.95)
    is_error = np.zeros(n, dtype=int)
    is_error[:100] = 1  # 20% wrong, all at 95% confidence
    df = pd.DataFrame({"_confidence": conf, "_is_error": is_error})
    result = calibration.confidence_analysis(df)
    assert result["n_high_confidence_errors"] == 100
    assert result["pct_errors_high_confidence"] == 1.0
