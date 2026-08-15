import io
import time
from pathlib import Path

import pandas as pd
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

ROOT = Path(__file__).resolve().parent.parent.parent


def test_excel_dataset_upload_is_parsed():
    df = pd.DataFrame({"a": [1, 2, 3, 4], "b": ["x", "y", "x", "y"], "target": [0, 1, 0, 1]})
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    buf.seek(0)

    r = client.post("/api/datasets/upload", files={"file": ("data.xlsx", buf, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_rows"] == 4
    assert "target" in body["columns"]
    assert body["profile"]["n_rows"] == 4


def test_unsupported_model_format_gives_clear_error():
    fake = io.BytesIO(b"not a real keras model")
    r = client.post("/api/models/upload", files={"file": ("model.h5", fake, "application/octet-stream")})
    assert r.status_code == 400
    assert "Keras" in r.json()["detail"]


def test_text_classification_full_flow():
    model_path = ROOT / "sample_models" / "text_classification.joblib"
    data_path = ROOT / "sample_data" / "text_classification_current.csv"
    if not (model_path.exists() and data_path.exists()):
        return  # sample artifacts not generated in this environment; skip gracefully

    with open(model_path, "rb") as f:
        r = client.post("/api/models/upload", files={"file": ("text_classification.joblib", f, "application/octet-stream")})
    assert r.status_code == 200, r.text
    model_id = r.json()["model_id"]

    with open(data_path, "rb") as f:
        r = client.post("/api/datasets/upload", files={"file": ("text_classification_current.csv", f, "text/csv")})
    dataset_id = r.json()["dataset_id"]

    r = client.post("/api/investigations", json={"model_id": model_id, "dataset_id": dataset_id, "target_column": "target"})
    investigation_id = r.json()["investigation_id"]

    for _ in range(60):
        r = client.get(f"/api/investigations/{investigation_id}")
        status = r.json()["status"]
        if status in ("completed", "failed"):
            break
        time.sleep(0.2)
    assert status == "completed", r.json().get("error")

    result = client.get(f"/api/investigations/{investigation_id}/results").json()
    assert 0.0 <= result["evaluation"]["accuracy"] <= 1.0
    assert result["dataset_info"]["n_features_used"] == 1  # only the text column
    assert len(result["findings"]) > 0
