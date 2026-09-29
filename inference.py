"""
Single-patient inference - SIH Hybrid QML project
=====================================================

Loads the artifacts saved by export_results.py and predicts for ONE patient.

INPUT: the 6 features in the SAME form as the quantum_ready CSV columns
(feature_1 ... feature_6, i.e. the PCA-reduced standardized values).
IMPORTANT: a dashboard user will not type PCA values. To take raw clinical
values (age, cholesterol, ...) the dashboard needs Person A's preprocessing
pipeline (scaler + PCA) exported and applied first. Ask Person A for it.

USAGE FROM PYTHON (e.g. in the dashboard backend):
    from inference import load_artifacts, predict_patient
    artifacts = load_artifacts("heart_disease")      # load once, reuse
    result = predict_patient("heart_disease", [0.1, -0.4, 1.2, 0.0, 0.3, -0.8], artifacts)

USAGE FROM THE TERMINAL (self-check):
    python inference.py heart_disease
  Re-predicts the first test patient from exported_results/<name>_predictions.csv
  and prints it next to the saved values. They should match.

Each call computes one quantum kernel row against every training patient, so
expect a few seconds per patient on the clean simulator.
"""

import os
import sys
import pickle
import numpy as np
import pandas as pd

from noise_robustness import clean_vqc_circuit
from quantum_kernel_svm import quantum_kernel

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exported_results")


def load_artifacts(name, out_dir=OUT_DIR):
    with open(os.path.join(out_dir, f"{name}_artifacts.pkl"), "rb") as f:
        return pickle.load(f)


def predict_patient(name, features, artifacts=None):
    a = artifacts or load_artifacts(name)
    features = np.asarray(features, dtype=float).reshape(1, -1)
    if features.shape[1] != len(a["feature_names"]):
        raise ValueError(f"Expected {len(a['feature_names'])} features, got {features.shape[1]}")

    x = a["scaler"].transform(features)[0]

    # Kernel SVM: similarity of this patient to every training patient
    k_row = np.array([[quantum_kernel(x, xt) for xt in a["X_train_scaled"]]])
    kernel_label = int(a["svc"].predict(k_row)[0])
    kernel_prob = float(a["svc"].predict_proba(k_row)[0, 1])

    # VQC: raw score in [-1, 1] (NOT a calibrated probability)
    vqc_score = float(clean_vqc_circuit(x, a["vqc_weights"]))

    return {
        "dataset": name,
        "kernel_svm": {"label": kernel_label, "prob_positive": round(kernel_prob, 4)},
        "vqc": {"label": int(vqc_score > 0), "score": round(vqc_score, 4)},
    }


if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "heart_disease"
    df = pd.read_csv(os.path.join(OUT_DIR, f"{name}_predictions.csv"))
    row = df.iloc[0]
    feats = [row[c] for c in df.columns if c.startswith("feature_")]
    print(f"Self-check on test patient 0 of {name} (true label {int(row['true_label'])}):")
    out = predict_patient(name, feats)
    print("  inference.py :", out)
    print("  saved values : kernel label", int(row["kernel_pred"]),
          "prob", row["kernel_prob_positive"],
          "| vqc label", int(row["vqc_pred"]), "score", row["vqc_score"])
    print("  (they should match, apart from tiny rounding)")
