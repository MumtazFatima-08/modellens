import time
from pathlib import Path

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = ROOT / "sample_models" / "healthy_classification.joblib"
DATA_PATH = ROOT / "sample_data" / "healthy_classification_current.csv"


def test_full_investigation_flow():
    assert MODEL_PATH.exists(), "Run scripts/generate_sample_models.py first"
    assert DATA_PATH.exists(), "Run scripts/generate_sample_data.py first"

    with open(MODEL_PATH, "rb") as f:
        r = client.post("/api/models/upload", files={"file": ("healthy_classification.joblib", f, "application/octet-stream")})
    assert r.status_code == 200, r.text
    model_info = r.json()
    assert model_info["task_type"] == "classification"
    model_id = model_info["model_id"]

    with open(DATA_PATH, "rb") as f:
        r = client.post("/api/datasets/upload", files={"file": ("healthy_classification_current.csv", f, "text/csv")})
    assert r.status_code == 200, r.text
    dataset_info = r.json()
    dataset_id = dataset_info["dataset_id"]
    assert "target" in dataset_info["columns"]

    r = client.post(
        "/api/investigations",
        json={"model_id": model_id, "dataset_id": dataset_id, "target_column": "target"},
    )
    assert r.status_code == 200, r.text
    investigation_id = r.json()["investigation_id"]

    for _ in range(60):
        r = client.get(f"/api/investigations/{investigation_id}")
        status = r.json()["status"]
        if status == "completed":
            break
        if status == "failed":
            raise AssertionError(f"Investigation failed: {r.json().get('error')}")
        time.sleep(0.2)
    else:
        raise AssertionError("Investigation did not complete in time")

    r = client.get(f"/api/investigations/{investigation_id}/results")
    assert r.status_code == 200
    result = r.json()

    # These must be REAL computed numbers, not placeholders.
    assert result["evaluation"]["task_type"] == "classification"
    assert 0.0 <= result["evaluation"]["accuracy"] <= 1.0
    assert result["evaluation"]["n_samples"] == 2000
    assert len(result["findings"]) > 0
    assert all(k in result["findings"][0] for k in ["severity", "title", "evidence", "interpretation", "recommendation"])

    r = client.get(f"/api/investigations/{investigation_id}/report", params={"format": "json"})
    assert r.status_code == 200
    r = client.get(f"/api/investigations/{investigation_id}/report", params={"format": "csv"})
    assert r.status_code == 200
    r = client.get(f"/api/investigations/{investigation_id}/report", params={"format": "pdf"})
    assert r.status_code == 200
    assert r.content[:4] == b"%PDF"


def test_incompatible_dataset_gives_useful_error():
    with open(MODEL_PATH, "rb") as f:
        r = client.post("/api/models/upload", files={"file": ("healthy_classification.joblib", f, "application/octet-stream")})
    model_id = r.json()["model_id"]

    import io
    bad_csv = b"a,b,target\n1,2,0\n3,4,1\n"
    r = client.post("/api/datasets/upload", files={"file": ("bad.csv", io.BytesIO(bad_csv), "text/csv")})
    dataset_id = r.json()["dataset_id"]

    r = client.post(
        "/api/investigations",
        json={"model_id": model_id, "dataset_id": dataset_id, "target_column": "target"},
    )
    investigation_id = r.json()["investigation_id"]

    for _ in range(30):
        r = client.get(f"/api/investigations/{investigation_id}")
        if r.json()["status"] in ("completed", "failed"):
            break
        time.sleep(0.2)

    assert r.json()["status"] == "failed"
    assert "expects" in r.json()["error"] or "features" in r.json()["error"]
