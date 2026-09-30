"""
FastAPI backend for the SIH Hybrid QML dashboard.

Run from E:\\SIH:
    uvicorn backend_main:app --reload --port 8000

Live prediction chain for a NEW patient:
    raw values (30 / 13 / 22 columns)
      -> Person A StandardScaler
            |-> classical model (LogReg / SVC / XGBoost)       <- scaled raw features, NO PCA
            |-> Person A PCA(6) -> angle_scaler (only if mode.json says so)
                   -> our quantum models (inference.predict_patient)

Label convention:
    breast_cancer uses scikit-learn's coding: 0 = malignant, 1 = benign, so "positive" =
    BENIGN. /predict and /comparison flip it so the dashboard always means "disease risk"
    (see POSITIVE_IS_DISEASE). The classical result is flipped too.
"""

import os
import json
import pickle

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR = os.path.join(BASE_DIR, "exported_results")
PREPROCESS_DIR = os.path.join(BASE_DIR, "preprocess")  # scaler/pca/angle per dataset + model_<name>.joblib
STATIC_RESULTS_PATH = os.path.join(BASE_DIR, "dashboard_static_results.json")
CLASSICAL_METRICS_PATH = os.path.join(PREPROCESS_DIR, "classical_metrics.json")
VALID_DATASETS = ["breast_cancer", "diabetes", "heart_disease", "parkinsons"]

# DEFAULT classical numbers (from the earlier handoff doc). Run classical_metrics.py to
# measure the saved models for real; its output (preprocess/classical_metrics.json)
# overrides these automatically. Model names below are the actual saved model types.
CLASSICAL_BASELINES = {
    "breast_cancer": {"model": "Logistic Regression",     "accuracy": 0.9649, "sensitivity": 0.9722, "specificity": 0.9524},
    "diabetes":      {"model": "Logistic Regression",     "accuracy": 0.7208, "sensitivity": 0.5556, "specificity": 0.8100},
    "heart_disease": {"model": "Support Vector Machine",  "accuracy": 0.8333, "sensitivity": 0.7500, "specificity": 0.9062},
    "parkinsons":    {"model": "XGBoost",                 "accuracy": 0.9487, "sensitivity": 1.0000, "specificity": 0.8000},
}

# Is label 1 the DISEASE class? (breast_cancer: no, label 1 = benign.)
POSITIVE_IS_DISEASE = {
    "breast_cancer": False,
    "diabetes": True,
    "heart_disease": True,
    "parkinsons": True,
}

app = FastAPI(title="Hybrid QML Disease Detection API")

from extract_api import router as extract_router
app.include_router(extract_router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev only - restrict this if you ever deploy publicly
    allow_methods=["*"],
    allow_headers=["*"],
)

_artifact_cache = {}
_preproc_cache = {}
_classical_model_cache = {}


# ---------------------------------------------------------------------------
# Classical model
# ---------------------------------------------------------------------------
_FRIENDLY_NAMES = {
    "LogisticRegression": "Logistic Regression",
    "SVC": "Support Vector Machine",
    "XGBClassifier": "XGBoost",
    "RandomForestClassifier": "Random Forest",
}


def friendly_model_name(model):
    n = type(model).__name__
    return _FRIENDLY_NAMES.get(n, n)


def classical_baseline(name):
    """Reported defaults, overridden by measured numbers if classical_metrics.py was run."""
    base = dict(CLASSICAL_BASELINES[name])
    if os.path.exists(CLASSICAL_METRICS_PATH):
        try:
            with open(CLASSICAL_METRICS_PATH) as f:
                measured = json.load(f).get(name)
            if measured:
                base.update({k: measured[k] for k in ("model", "accuracy", "sensitivity", "specificity") if k in measured})
        except Exception:
            pass
    return base


def _classical_model_path(name):
    candidates = [
        os.path.join(PREPROCESS_DIR, f"model_{name}.joblib"),
        os.path.join(PREPROCESS_DIR, name, f"model_{name}.joblib"),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError("No classical model file. Looked for: " + " | ".join(candidates))


def load_classical_model(name):
    if name in _classical_model_cache:
        return _classical_model_cache[name]
    path = _classical_model_path(name)
    model = joblib.load(path)  # Parkinson's needs `pip install xgboost`
    _classical_model_cache[name] = (model, path)
    return _classical_model_cache[name]


def predict_classical(name, x_scaled):
    """x_scaled: Person A's StandardScaler output, shape (1, n_raw). No PCA."""
    model, _ = load_classical_model(name)
    expected = getattr(model, "n_features_in_", None)
    if expected is not None and x_scaled.shape[1] != expected:
        raise ValueError(
            f"classical model for '{name}' expects {expected} features, got {x_scaled.shape[1]}"
        )
    proba = model.predict_proba(x_scaled)[0]
    return {
        "label": int(proba[1] > 0.5),
        "prob_positive": round(float(proba[1]), 4),
        "model": friendly_model_name(model),
    }


@app.get("/datasets/{name}/classical-status")
def classical_status(name: str):
    if name not in VALID_DATASETS:
        raise HTTPException(status_code=400, detail=f"Unknown dataset '{name}'")
    try:
        model, path = load_classical_model(name)
        return {
            "available": True,
            "model": friendly_model_name(model),
            "file": os.path.basename(path),
            "n_features_in": getattr(model, "n_features_in_", None),
        }
    except Exception as e:
        return {"available": False, "reason": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------------------
# Helpers for exported results / quantum artifacts
# ---------------------------------------------------------------------------
def dataset_dir_ready(name):
    return os.path.exists(os.path.join(EXPORT_DIR, f"{name}_predictions.csv"))


def load_predictions(name):
    path = os.path.join(EXPORT_DIR, f"{name}_predictions.csv")
    if not os.path.exists(path):
        raise HTTPException(
            status_code=404,
            detail=f"No exported results for '{name}' yet. Run: python -u export_results.py {name}",
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


def get_preproc(name):
    """Person A's fitted scaler + PCA (+ angle scaler and the verified mode)."""
    if name in _preproc_cache:
        return _preproc_cache[name]
    folder = os.path.join(PREPROCESS_DIR, name)
    scaler_path = os.path.join(folder, "scaler.joblib")
    pca_path = os.path.join(folder, "pca.joblib")
    if not (os.path.exists(scaler_path) and os.path.exists(pca_path)):
        raise HTTPException(
            status_code=404,
            detail=f"Live prediction isn't available for '{name}' yet: "
                   f"expected {scaler_path} and {pca_path}.",
        )

    pre = {
        "scaler": joblib.load(scaler_path),
        "pca": joblib.load(pca_path),
        "angle": None,
        "apply_angle": None,
    }
    angle_path = os.path.join(folder, "angle_scaler.joblib")
    if os.path.exists(angle_path):
        pre["angle"] = joblib.load(angle_path)
    mode_path = os.path.join(folder, "mode.json")
    if os.path.exists(mode_path):
        with open(mode_path) as f:
            pre["apply_angle"] = json.load(f).get("apply_angle_scaler")

    # don't cache until the mode is known, so running verify_preprocess.py
    # takes effect without a stale cache
    if pre["apply_angle"] is not None:
        _preproc_cache[name] = pre
    return pre


@app.get("/datasets/{name}/preprocess-status")
def preprocess_status(name: str):
    """Debug helper: what the live-prediction chain will use for this dataset."""
    if name not in VALID_DATASETS:
        raise HTTPException(status_code=400, detail=f"Unknown dataset '{name}'")
    pre = get_preproc(name)
    return {
        "raw_inputs_expected": int(pre["scaler"].n_features_in_),
        "pca_components": int(pre["pca"].n_components_),
        "angle_scaler_present": pre["angle"] is not None,
        "apply_angle_scaler": pre["apply_angle"],
        "positive_label_is_disease": POSITIVE_IS_DISEASE.get(name, True),
    }


def _flip_to_disease_positive(result):
    """Turn 'positive = benign' results into 'positive = malignant' (breast_cancer)."""
    for key in ("classical", "kernel_svm", "vqc"):
        m = result.get(key)
        if not m:
            continue
        if "label" in m:
            m["label"] = 1 - int(m["label"])
        if m.get("prob_positive") is not None:
            m["prob_positive"] = round(1.0 - float(m["prob_positive"]), 4)
        if m.get("score") is not None:
            m["score"] = round(-float(m["score"]), 4)
    return result


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.get("/datasets")
def list_datasets():
    return {"datasets": [{"name": n, "ready": dataset_dir_ready(n)} for n in VALID_DATASETS]}


@app.get("/datasets/{name}/metrics")
def dataset_metrics(name: str):
    if name not in VALID_DATASETS:
        raise HTTPException(status_code=400, detail=f"Unknown dataset '{name}'")
    return load_metrics(name)


@app.get("/datasets/{name}/comparison")
def dataset_comparison(name: str):
    """Classical + Kernel SVM + VQC, side by side (sensitivity = share of DISEASE cases caught)."""
    if name not in VALID_DATASETS:
        raise HTTPException(status_code=400, detail=f"Unknown dataset '{name}'")
    quantum = load_metrics(name)
    classical = classical_baseline(name)

    def row(model_name, kind, m):
        sens, spec = m["sensitivity"], m["specificity"]
        if not POSITIVE_IS_DISEASE.get(name, True):
            sens, spec = spec, sens  # positive class was benign: swap so sensitivity = malignant caught
        return {
            "name": model_name,
            "kind": kind,
            "accuracy": m["accuracy"],
            "sensitivity": sens,
            "specificity": spec,
        }

    return {
        "dataset": name,
        "n_test": quantum["n_test"],
        "models": [
            row(classical["model"] + " (classical)", "classical", classical),
            row("Quantum Kernel SVM", "kernel_svm", quantum["kernel_svm"]),
            row("VQC (seed 42)", "vqc", quantum["vqc_seed42"]),
        ],
    }


@app.get("/comparison/all")
def comparison_all():
    out = []
    for name in VALID_DATASETS:
        if dataset_dir_ready(name):
            out.append(dataset_comparison(name))
    return {"datasets": out}


@app.get("/datasets/{name}/patients")
def list_patients(name: str):
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
        ..., min_length=1,
        description="Raw clinical values, in the same column order used in training.",
    )


@app.post("/datasets/{name}/predict")
def predict_new_patient(name: str, body: PredictRequest):
    """
    Live inference on RAW values. Slow (a few seconds): recomputes a quantum
    kernel row + one VQC evaluation.
    """
    if name not in VALID_DATASETS:
        raise HTTPException(status_code=400, detail=f"Unknown dataset '{name}'")

    from inference import predict_patient  # lazy import (PennyLane)

    pre = get_preproc(name)
    scaler, pca = pre["scaler"], pre["pca"]

    x = np.asarray(body.features, dtype=float).reshape(1, -1)
    if x.shape[1] != scaler.n_features_in_:
        raise HTTPException(
            status_code=422,
            detail=f"Expected {scaler.n_features_in_} values, got {x.shape[1]}.",
        )

    if pre["apply_angle"] is None:
        raise HTTPException(
            status_code=409,
            detail="Preprocessing mode not verified yet. Run `python verify_preprocess.py` "
                   "from E:\\SIH, then restart uvicorn.",
        )

    x_scaled = scaler.transform(x)          # Person A StandardScaler (used by classical AND quantum path)
    z = pca.transform(x_scaled)
    if pre["apply_angle"]:
        if pre["angle"] is None:
            raise HTTPException(status_code=404, detail=f"angle_scaler.joblib missing for '{name}'.")
        z = pre["angle"].transform(z)
    x6 = z[0]

    try:
        result = predict_patient(name, x6.tolist(), get_artifacts(name))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    # classical model: never allowed to break the quantum result
    classical, classical_error = None, None
    try:
        classical = predict_classical(name, x_scaled)
    except Exception as e:
        classical_error = f"{type(e).__name__}: {e}"
    result["classical"] = classical
    if classical_error:
        result["classical_error"] = classical_error

    if not POSITIVE_IS_DISEASE.get(name, True):
        result = _flip_to_disease_positive(result)
    return result


@app.get("/results/summary")
def results_summary():
    if not os.path.exists(STATIC_RESULTS_PATH):
        raise HTTPException(status_code=404, detail="dashboard_static_results.json not found")
    with open(STATIC_RESULTS_PATH) as f:
        return json.load(f)


@app.get("/charts/{filename}")
def get_chart(filename: str):
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