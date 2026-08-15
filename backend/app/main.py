from __future__ import annotations

import logging
import shutil
import traceback
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app import storage
from app.ml_engine import adapters, dataset_profile
from app.ml_engine.orchestrator import InvestigationError, run_investigation
from app.report import build_csv_report, build_json_report, build_pdf_report

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("modellens")

app = FastAPI(title="ModelLens API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SAMPLE_MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "sample_models"
SAMPLE_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "sample_data"


@app.get("/api/health")
def health():
    return {"status": "ok"}


# ---------- Model upload ----------

@app.post("/api/models/upload")
async def upload_model(file: UploadFile = File(...)):
    lower = file.filename.lower()
    unsupported_hints = {
        ".h5": "Keras/TensorFlow (.h5)",
        ".keras": "Keras (.keras)",
        ".pt": "PyTorch (.pt)",
        ".pth": "PyTorch (.pth)",
        ".onnx": "ONNX (.onnx)",
        ".pb": "TensorFlow SavedModel (.pb)",
    }
    for ext, label in unsupported_hints.items():
        if lower.endswith(ext):
            raise HTTPException(
                400,
                f"{label} models are not yet supported — ModelLens currently investigates "
                "scikit-learn-compatible estimators (including XGBoost/LightGBM sklearn "
                "wrappers) saved as .pkl or .joblib. Support for other frameworks is on the "
                "roadmap; see README.md > Limitations.",
            )
    if not lower.endswith((".pkl", ".joblib")):
        raise HTTPException(400, "Model file must be .pkl or .joblib")
    raw = await file.read()
    model_id, path = storage.save_model_file(raw, file.filename)
    try:
        loaded = adapters.load_model(path)
    except adapters.UnsupportedModelError as e:
        path.unlink(missing_ok=True)
        raise HTTPException(400, str(e))
    return {
        "model_id": model_id,
        "filename": file.filename,
        "task_type": loaded.info.task_type,
        "estimator_class": loaded.info.estimator_class,
        "n_features_expected": loaded.info.n_features_expected,
        "feature_names_expected": loaded.info.feature_names_expected,
        "classes": [str(c) for c in loaded.info.classes] if loaded.info.classes else None,
        "supports_proba": loaded.info.supports_proba,
    }


# ---------- Dataset upload ----------

@app.post("/api/datasets/upload")
async def upload_dataset(file: UploadFile = File(...)):
    lower = file.filename.lower()
    if not lower.endswith((".csv", ".xlsx", ".xls")):
        raise HTTPException(400, "Dataset file must be .csv, .xlsx, or .xls")
    raw = await file.read()

    if lower.endswith((".xlsx", ".xls")):
        # Store a normalized CSV on disk so the rest of the pipeline (which
        # is CSV-based end to end) doesn't need to special-case Excel later.
        import io as _io
        try:
            df = pd.read_excel(_io.BytesIO(raw))
        except Exception as e:
            raise HTTPException(400, f"Could not parse Excel file: {e}")
        dataset_id = storage.new_id()
        path = storage.DATASETS_DIR / f"{dataset_id}.csv"
        df.to_csv(path, index=False)
    else:
        dataset_id, path = storage.save_dataset_file(raw, file.filename)
        try:
            df = pd.read_csv(path)
        except Exception as e:
            path.unlink(missing_ok=True)
            raise HTTPException(400, f"Could not parse CSV: {e}")

    profile = dataset_profile.profile_dataset(df)
    return {
        "dataset_id": dataset_id,
        "filename": file.filename,
        "n_rows": int(len(df)),
        "columns": list(df.columns),
        "preview": df.head(5).to_dict(orient="records"),
        "profile": profile,
    }


# ---------- Samples ----------

CASES = [
    {"id": "healthy_classification", "label": "Healthy Classification Model", "task": "classification"},
    {"id": "weak_classification", "label": "Weak Classification Model", "task": "classification"},
    {"id": "imbalanced_classification", "label": "Imbalanced Classification", "task": "classification"},
    {"id": "distribution_shift", "label": "Distribution Shift", "task": "classification"},
    {"id": "error_prone", "label": "Error-Prone Model (clustered errors)", "task": "classification"},
    {"id": "text_classification", "label": "Text Classification (TF-IDF, no numeric features)", "task": "classification"},
    {"id": "regression", "label": "Regression Model", "task": "regression"},
]


@app.get("/api/samples")
def list_samples():
    available = []
    for c in CASES:
        model_path = SAMPLE_MODELS_DIR / f"{c['id']}.joblib"
        data_path = SAMPLE_DATA_DIR / f"{c['id']}_current.csv"
        ref_path = SAMPLE_DATA_DIR / f"{c['id']}_reference.csv"
        available.append(
            {
                **c,
                "ready": model_path.exists() and data_path.exists(),
                "has_reference": ref_path.exists(),
            }
        )
    return {"cases": available}


class SampleLoadRequest(BaseModel):
    case_id: str


@app.post("/api/samples/load")
def load_sample(req: SampleLoadRequest):
    case = next((c for c in CASES if c["id"] == req.case_id), None)
    if not case:
        raise HTTPException(404, f"Unknown sample case '{req.case_id}'")

    model_path = SAMPLE_MODELS_DIR / f"{case['id']}.joblib"
    data_path = SAMPLE_DATA_DIR / f"{case['id']}_current.csv"
    ref_path = SAMPLE_DATA_DIR / f"{case['id']}_reference.csv"

    if not model_path.exists() or not data_path.exists():
        raise HTTPException(
            404,
            "Sample artifacts not generated yet. Run "
            "'python scripts/generate_sample_models.py' first.",
        )

    model_id = storage.new_id()
    dest_model = storage.MODELS_DIR / f"{model_id}.joblib"
    shutil.copy(model_path, dest_model)

    dataset_id = storage.new_id()
    dest_data = storage.DATASETS_DIR / f"{dataset_id}.csv"
    shutil.copy(data_path, dest_data)

    reference_dataset_id = None
    if ref_path.exists():
        reference_dataset_id = storage.new_id()
        dest_ref = storage.DATASETS_DIR / f"{reference_dataset_id}.csv"
        shutil.copy(ref_path, dest_ref)

    df = pd.read_csv(data_path)
    target_column = "target"

    return {
        "model_id": model_id,
        "dataset_id": dataset_id,
        "reference_dataset_id": reference_dataset_id,
        "target_column": target_column,
        "columns": list(df.columns),
    }


# ---------- Investigations ----------

class CreateInvestigationRequest(BaseModel):
    model_id: str
    dataset_id: str
    target_column: str
    feature_columns: Optional[list[str]] = None
    reference_dataset_id: Optional[str] = None


def _run_job(investigation_id: str, req: CreateInvestigationRequest):
    model_path = storage.model_path_for(req.model_id)
    dataset_path = storage.dataset_path_for(req.dataset_id)
    reference_path = (
        storage.dataset_path_for(req.reference_dataset_id) if req.reference_dataset_id else None
    )

    if model_path is None or dataset_path is None:
        storage.update_job(investigation_id, status="failed", error="Model or dataset not found.", progress=0)
        return

    def progress_cb(pct, msg):
        storage.update_job(investigation_id, status="running", progress=pct, message=msg)

    try:
        storage.update_job(investigation_id, status="running", progress=0, message="Starting")
        result = run_investigation(
            model_path=model_path,
            dataset_path=dataset_path,
            target_column=req.target_column,
            feature_columns=req.feature_columns,
            reference_dataset_path=reference_path,
            progress_cb=progress_cb,
        )
        storage.save_result(investigation_id, result)
        storage.update_job(investigation_id, status="completed", progress=100, message="Done")
    except (InvestigationError, adapters.ModelDatasetMismatchError) as e:
        storage.update_job(investigation_id, status="failed", error=str(e), progress=0)
    except Exception as e:
        logger.error("Investigation failed: %s\n%s", e, traceback.format_exc())
        storage.update_job(
            investigation_id, status="failed", error=f"Unexpected error: {e}", progress=0
        )


@app.post("/api/investigations")
def create_investigation(req: CreateInvestigationRequest, background_tasks: BackgroundTasks):
    investigation_id = storage.new_id()
    storage.register_job(
        investigation_id,
        {"status": "queued", "progress": 0, "message": "Queued", "error": None},
    )
    background_tasks.add_task(_run_job, investigation_id, req)
    return {"investigation_id": investigation_id, "status": "queued"}


@app.get("/api/investigations/{investigation_id}")
def get_investigation_status(investigation_id: str):
    job = storage.get_job(investigation_id)
    if job is None:
        raise HTTPException(404, "Investigation not found")
    return {"investigation_id": investigation_id, **job}


@app.get("/api/investigations/{investigation_id}/results")
def get_investigation_results(investigation_id: str):
    job = storage.get_job(investigation_id)
    if job is None:
        raise HTTPException(404, "Investigation not found")
    if job["status"] != "completed":
        raise HTTPException(409, f"Investigation is not complete (status={job['status']})")
    result = storage.load_result(investigation_id)
    if result is None:
        raise HTTPException(404, "Result not found")
    return result


@app.get("/api/investigations/{investigation_id}/report")
def get_investigation_report(investigation_id: str, format: str = "json"):
    from fastapi.responses import Response

    job = storage.get_job(investigation_id)
    if job is None or job["status"] != "completed":
        raise HTTPException(409, "Investigation is not complete")
    result = storage.load_result(investigation_id)

    if format == "json":
        content = build_json_report(investigation_id, result)
        return Response(content=content, media_type="application/json",
                         headers={"Content-Disposition": f"attachment; filename=modellens_report_{investigation_id}.json"})
    elif format == "csv":
        content = build_csv_report(result)
        return Response(content=content, media_type="text/csv",
                         headers={"Content-Disposition": f"attachment; filename=modellens_report_{investigation_id}.csv"})
    elif format == "pdf":
        content = build_pdf_report(investigation_id, result)
        return Response(content=content, media_type="application/pdf",
                         headers={"Content-Disposition": f"attachment; filename=modellens_report_{investigation_id}.pdf"})
    else:
        raise HTTPException(400, "format must be one of: json, csv, pdf")
