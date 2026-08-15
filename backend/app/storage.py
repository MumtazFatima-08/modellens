"""
Local storage layer.

NOTE (documented limitation): this uses local disk for files and an
in-memory dict for job/investigation status, which is appropriate for a
locally-run single-process app but is NOT suitable for multi-worker
production deployment as-is. A production deployment would swap this for
object storage (S3/GCS) + a real job queue (Celery/RQ) + a database
(Postgres) behind the same interface.
"""
from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Any, Optional

BASE_DIR = Path(__file__).resolve().parent.parent / "storage"
MODELS_DIR = BASE_DIR / "models"
DATASETS_DIR = BASE_DIR / "datasets"
INVESTIGATIONS_DIR = BASE_DIR / "investigations"

for d in (MODELS_DIR, DATASETS_DIR, INVESTIGATIONS_DIR):
    d.mkdir(parents=True, exist_ok=True)

_lock = Lock()
_jobs: dict[str, dict] = {}


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def save_model_file(raw_bytes: bytes, original_filename: str) -> tuple[str, Path]:
    model_id = new_id()
    suffix = Path(original_filename).suffix.lower()
    path = MODELS_DIR / f"{model_id}{suffix}"
    path.write_bytes(raw_bytes)
    return model_id, path


def save_dataset_file(raw_bytes: bytes, original_filename: str) -> tuple[str, Path]:
    dataset_id = new_id()
    path = DATASETS_DIR / f"{dataset_id}.csv"
    path.write_bytes(raw_bytes)
    return dataset_id, path


def model_path_for(model_id: str) -> Optional[Path]:
    matches = list(MODELS_DIR.glob(f"{model_id}.*"))
    return matches[0] if matches else None


def dataset_path_for(dataset_id: str) -> Optional[Path]:
    path = DATASETS_DIR / f"{dataset_id}.csv"
    return path if path.exists() else None


def register_job(job_id: str, meta: dict) -> None:
    with _lock:
        _jobs[job_id] = meta


def update_job(job_id: str, **kwargs) -> None:
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)


def get_job(job_id: str) -> Optional[dict]:
    with _lock:
        return dict(_jobs[job_id]) if job_id in _jobs else None


def save_result(investigation_id: str, result: dict) -> None:
    path = INVESTIGATIONS_DIR / f"{investigation_id}.json"
    path.write_text(json.dumps(result, indent=2, default=str))


def load_result(investigation_id: str) -> Optional[dict]:
    path = INVESTIGATIONS_DIR / f"{investigation_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())
