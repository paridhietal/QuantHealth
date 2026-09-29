"""
FastAPI backend for the SIH Hybrid QML dashboard
====================================================

WHAT THIS WRAPS (all built earlier, unchanged):
  - exported_results/<name>_predictions.csv   (from export_results.py)
  - exported_results/<name>_metrics.json      (from export_results.py)
  - exported_results/<name>_artifacts.pkl     (from export_results.py, used by inference.py)
  - dashboard_static_results.json             (benchmark, seed check, noise, calibration, MI)
  - vqc_noise_robustness.png, calibration_heart_disease.png (chart images)

RUN IT (same folder as everything above, plus noise_robustness.py,
quantum_kernel_svm.py, inference.py):
    pip install fastapi uvicorn --break-system-packages   (if not already installed)
    uvicorn backend_main:app --reload --port 8000

Then open http://localhost:8000/docs to see every endpoint and try them by
hand before wiring up Next.js.

ENDPOINTS:
  GET  /datasets                         which datasets have exported files
  GET  /datasets/{name}/metrics          accuracy/sensitivity/specificity (from export)
  GET  /datasets/{name}/patients         list of test patients (id, true label, correctness)
  GET  /datasets/{name}/patients/{id}    one patient's saved prediction (fast, no recompute)
  POST /datasets/{name}/predict          run inference on a NEW feature vector (slow: recomputes
                                         the quantum kernel row + circuit; a few seconds)
  GET  /results/summary                 the whole dashboard_static_results.json
  GET  /charts/{filename}                serves a chart PNG by filename

CORS: wide open (allow all origins) for local hackathon development. Tighten
this before deploying anywhere public.
"""

import os
import json
import pickle

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR = os.path.join(BASE_DIR, "exported_results")
STATIC_RESULTS_PATH = os.path.join(BASE_DIR, "dashboard_static_results.json")
VALID_DATASETS = ["breast_cancer", "diabetes", "heart_disease", "parkinsons"]

app = FastAPI(title="Hybrid QML Disease Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only - restrict this if you ever deploy publicly
    allow_methods=["*"],
    allow_headers=["*"],
)

# in-memory cache so we don't re-read CSVs/pickles on every request
_artifact_cache = {}


def dataset_dir_ready(name):
    return os.path.exists(os.path.join(EXPORT_DIR, f"{name}_predictions.csv"))


def load_predictions(name):
    path = os.path.join(EXPORT_DIR, f"{name}_predictions.csv")
    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail=f"No exported results for '{name}' yet. "
                   f"Run: python -u export_results.py {name}",
        )
    return pd.read_csv(path, index_col="patient_id")


def load_metrics(name):
    path = os.path.join(EXPORT_DIR, f"{name}_metrics.json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"No metrics exported for '{name}' yet.")
    with open(path) as f:
        return json.load(f)


def get_artifacts(name):
    if name in _artifact_cache:
        return _artifact_cache[name]
    path = os.path.join(EXPORT_DIR, f"{name}_artifacts.pkl")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"No artifacts exported for '{name}' yet.")
    with open(path, "rb") as f:
        artifacts = pickle.load(f)
    _artifact_cache[name] = artifacts
    return artifacts


@app.get("/datasets")
def list_datasets():
    """Which datasets are ready to serve, based on what's been exported so far."""
    return {
        "datasets": [
            {"name": n, "ready": dataset_dir_ready(n)}
            for n in VALID_DATASETS
        ]
    }


@app.get("/datasets/{name}/metrics")
def dataset_metrics(name: str):
    if name not in VALID_DATASETS:
        raise HTTPException(status_code=400, detail=f"Unknown dataset '{name}'")
    return load_metrics(name)


@app.get("/datasets/{name}/patients")
def list_patients(name: str):
    """
    Lightweight list for a picker UI: id, true label, and whether each
    model got it right. Does NOT include the raw feature values (use the
    single-patient endpoint for that) to keep this response small.
    """
    if name not in VALID_DATASETS:
        raise HTTPException(status_code=400, detail=f"Unknown dataset '{name}'")
    df = load_predictions(name)
    patients = [
        {
            "patient_id": int(idx),
            "true_label": int(row["true_label"]),
            "kernel_correct": bool(row["kernel_correct"]),
            "vqc_correct": bool(row["vqc_correct"]),
        }
        for idx, row in df.iterrows()
    ]
    return {"dataset": name, "n_patients": len(patients), "patients": patients}


@app.get("/datasets/{name}/patients/{patient_id}")
def get_patient(name: str, patient_id: int):
    """One test patient's saved prediction - instant, no recomputation."""
    if name not in VALID_DATASETS:
        raise HTTPException(status_code=400, detail=f"Unknown dataset '{name}'")
    df = load_predictions(name)
    if patient_id not in df.index:
        raise HTTPException(status_code=404, detail=f"No patient {patient_id} in '{name}'")
    row = df.loc[patient_id]
    feature_cols = [c for c in df.columns if c.startswith("feature_")]
    return {
        "dataset": name,
        "patient_id": patient_id,
        "true_label": int(row["true_label"]),
        "features": {c: float(row[c]) for c in feature_cols},
        "kernel_svm": {
            "label": int(row["kernel_pred"]),
            "prob_positive": float(row["kernel_prob_positive"]),
            "correct": bool(row["kernel_correct"]),
        },
        "vqc": {
            "label": int(row["vqc_pred"]),
            "score": float(row["vqc_score"]),
            "correct": bool(row["vqc_correct"]),
            "note": "VQC score is in [-1, 1], not a calibrated probability. "
                    "Show the score, not a risk percentage.",
        },
    }


class PredictRequest(BaseModel):
    features: list[float] = Field(
        ..., min_length=6, max_length=6,
        description="The 6 PCA features, same form as the quantum_ready CSV columns.",
    )


@app.post("/datasets/{name}/predict")
def predict_new_patient(name: str, body: PredictRequest):
    """
    Runs REAL inference on a feature vector that wasn't in the saved test
    set. This is slow (recomputes a quantum kernel row against every
    training patient plus one VQC circuit evaluation) - expect a few
    seconds, not milliseconds. For the demo, prefer the saved-patient
    endpoint above and only use this for a live "type in some numbers" bit.
    """
    if name not in VALID_DATASETS:
        raise HTTPException(status_code=400, detail=f"Unknown dataset '{name}'")

    # imported lazily so the server can still start up even if PennyLane
    # isn't installed for datasets you haven't touched yet
    from inference import predict_patient

    artifacts = get_artifacts(name)
    try:
        result = predict_patient(name, body.features, artifacts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return result


@app.get("/results/summary")
def results_summary():
    """Everything already measured: benchmark table, seed check, noise, calibration, MI."""
    if not os.path.exists(STATIC_RESULTS_PATH):
        raise HTTPException(status_code=404, detail="dashboard_static_results.json not found")
    with open(STATIC_RESULTS_PATH) as f:
        return json.load(f)


@app.get("/charts/{filename}")
def get_chart(filename: str):
    """Serves a chart PNG. Only these two exact filenames are allowed."""
    allowed = {"vqc_noise_robustness.png", "calibration_heart_disease.png"}
    if filename not in allowed:
        raise HTTPException(status_code=404, detail="Unknown chart file")
    path = os.path.join(BASE_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail=f"{filename} not found next to backend_main.py")
    return FileResponse(path, media_type="image/png")


@app.get("/")
def root():
    return {
        "message": "Hybrid QML Disease Detection API is running.",
        "docs": "/docs",
        "datasets": VALID_DATASETS,
    }