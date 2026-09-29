"""
predict_api.py — live prediction from raw clinical inputs
=============================================================

WHAT THIS DOES:
Takes the values a person types into the frontend form (age, cholesterol,
glucose, ...) and turns them into a prediction from both the quantum kernel
SVM and the VQC, using the same trained models export_results.py already
produced.

THE MISSING PIECE — READ THIS FIRST:
inference.py's predict_patient() expects PCA-reduced, standardized features
(the feature_1..6 columns in your CSVs), not raw clinical values. Turning
"age=54, cholesterol=230, ..." into those 6 numbers requires the EXACT same
StandardScaler + PCA that was fit when the quantum_ready CSVs were built —
that fitting happens inside noise_robustness.load_dataset(), which we don't
have visibility into from the dashboard side.

So this file expects a file named <dataset>_raw_preprocess.pkl in
exported_results/, containing:
    {
        "feature_names": [...],   # raw feature order, matching lib/patientFields.ts
        "scaler": <fitted StandardScaler>,
        "pca": <fitted PCA, n_components=6>,
    }

Ask Person A to add ~5 lines to load_dataset() (or wherever the scaler/PCA
are fit) to pickle these three objects out. Until that file exists, this
endpoint returns a clear 501 error instead of a wrong prediction — it will
NEVER silently guess.

USAGE:
    uvicorn predict_api:app --reload
Mount this router in your existing FastAPI app, or run standalone for testing.
"""

import os
import pickle
from typing import Dict, Optional

import numpy as np
from fastapi import APIRouter, FastAPI, HTTPException
from pydantic import BaseModel

from inference import load_artifacts, predict_patient

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exported_results")
VALID = ["breast_cancer", "diabetes", "heart_disease", "parkinsons"]

router = APIRouter()
_raw_cache: Dict[str, dict] = {}
_artifact_cache: Dict[str, dict] = {}


class PredictRequest(BaseModel):
    name: Optional[str] = None
    age: Optional[float] = None
    features: Dict[str, float]


def _load_raw_preprocess(dataset: str) -> dict:
    if dataset in _raw_cache:
        return _raw_cache[dataset]
    path = os.path.join(OUT_DIR, f"{dataset}_raw_preprocess.pkl")
    if not os.path.exists(path):
        raise HTTPException(
            status_code=501,
            detail=(
                f"No raw-feature preprocessing found for '{dataset}'. "
                f"Expected {path}. This needs Person A to export the "
                f"StandardScaler + PCA used before the quantum_ready CSVs "
                f"were built — see the comment at the top of predict_api.py."
            ),
        )
    with open(path, "rb") as f:
        obj = pickle.load(f)
    for key in ("feature_names", "scaler", "pca"):
        if key not in obj:
            raise HTTPException(
                status_code=500,
                detail=f"{dataset}_raw_preprocess.pkl is missing '{key}'.",
            )
    _raw_cache[dataset] = obj
    return obj


def _load_model_artifacts(dataset: str) -> dict:
    if dataset in _artifact_cache:
        return _artifact_cache[dataset]
    artifacts = load_artifacts(dataset, out_dir=OUT_DIR)
    _artifact_cache[dataset] = artifacts
    return artifacts


def raw_to_quantum_features(dataset: str, raw: Dict[str, float]) -> list:
    """Raw clinical dict -> the 6 PCA-reduced values inference.py expects."""
    pre = _load_raw_preprocess(dataset)
    missing = [k for k in pre["feature_names"] if k not in raw]
    if missing:
        raise HTTPException(
            status_code=422,
            detail=f"Missing required fields for {dataset}: {missing}",
        )
    ordered = np.array([[float(raw[k]) for k in pre["feature_names"]]])
    scaled = pre["scaler"].transform(ordered)
    reduced = pre["pca"].transform(scaled)
    return reduced[0].tolist()


@router.post("/predict/{dataset}")
def predict(dataset: str, body: PredictRequest):
    if dataset not in VALID:
        raise HTTPException(status_code=404, detail=f"Unknown dataset '{dataset}'.")

    quantum_features = raw_to_quantum_features(dataset, body.features)
    artifacts = _load_model_artifacts(dataset)

    try:
        result = predict_patient(dataset, quantum_features, artifacts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}")

    return {
        "kernel_svm": {
            "label": result["kernel_svm"]["label"],
            "prob_positive": result["kernel_svm"]["prob_positive"],
        },
        "vqc": {
            "label": result["vqc"]["label"],
            "score": result["vqc"]["score"],
        },
    }


# Standalone run: `uvicorn predict_api:app --reload`
app = FastAPI()
app.include_router(router)