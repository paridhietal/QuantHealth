"""
Export results for the dashboard - SIH Hybrid QML project
=============================================================

WHY THIS EXISTS:
The quantum kernel step takes minutes, so a live dashboard cannot recompute
it on stage. Run this ONCE per dataset. It trains the two quantum models,
scores the test patients, and saves everything to the exported_results/
folder so the dashboard can just load files (and so single-patient inference
works later through inference.py).

WHAT IT SAVES (per dataset, into exported_results/):
  <name>_predictions.csv   one row per test patient: true label, the 6 input
                           features, Kernel SVM label + probability, VQC score
                           + label, and whether each was correct
  <name>_artifacts.pkl     scaler, scaled training data, fitted SVC, VQC
                           weights (needed by inference.py)
  <name>_vqc_weights.npy   the trained VQC weights on their own
  <name>_kernel_cache.npz  the two kernel matrices (so reruns skip the slow
                           step; DELETE this file if the CSVs ever change)
  <name>_metrics.json      accuracy / sensitivity / specificity per model
  metrics_all.json         all datasets exported so far, merged

HOW TO RUN (same folder as noise_robustness.py, quantum_kernel_svm.py, CSVs):
    python -u export_results.py heart_disease
    python -u export_results.py parkinsons
    python -u export_results.py all        (slow: breast_cancer / diabetes are big)

NOTES FOR THE DASHBOARD:
  - The VQC is the seed-42 model, the same as the main benchmark table.
  - The VQC gives a SCORE in [-1, 1], not a calibrated probability. Our
    calibration test showed its probabilities are not informative, so show
    the score and label only, not a "risk %".
  - The Kernel SVM probability comes from scikit-learn's built-in scaling and
    is more informative. Its label is the SVM decision label, which matches
    the benchmark accuracy; the probability at a 0.5 cutoff can differ
    slightly for a few borderline patients.
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from pennylane import numpy as pnp
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix

from noise_robustness import load_dataset, train_vqc, clean_vqc_circuit
from quantum_kernel_svm import compute_kernel_matrix

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exported_results")
RANDOM_STATE = 42
VALID = ["breast_cancer", "diabetes", "heart_disease", "parkinsons"]


def metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "sensitivity": float(tp / (tp + fn)) if (tp + fn) else None,
        "specificity": float(tn / (tn + fp)) if (tn + fp) else None,
    }


def kernel_matrices(name, Xs_train, Xs_test):
    cache = os.path.join(OUT_DIR, f"{name}_kernel_cache.npz")
    if os.path.exists(cache):
        print("  loaded cached kernel matrices", flush=True)
        d = np.load(cache)
        return d["K_train"], d["K_test"]
    print(f"  computing kernel matrices ({len(Xs_train)}x{len(Xs_train)} + "
          f"{len(Xs_test)}x{len(Xs_train)}) - takes a few minutes...", flush=True)
    K_train = compute_kernel_matrix(Xs_train, Xs_train)
    K_test = compute_kernel_matrix(Xs_test, Xs_train)
    np.savez(cache, K_train=K_train, K_test=K_test)
    return K_train, K_test


def export_dataset(name):
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"\n{'='*60}\nExporting: {name}\n{'='*60}", flush=True)

    X_train, X_test, y_train, y_test = load_dataset(name)
    y_train = np.asarray(y_train).astype(int)
    y_test = np.asarray(y_test).astype(int)

    # scale into [-pi, pi], fit on train only (same as every other script)
    scaler = MinMaxScaler(feature_range=(-np.pi, np.pi)).fit(X_train)
    Xs_train = scaler.transform(X_train)
    Xs_test = scaler.transform(X_test)

    # ---- Quantum kernel SVM ----
    K_train, K_test = kernel_matrices(name, Xs_train, Xs_test)
    svc = SVC(kernel="precomputed", probability=True, random_state=RANDOM_STATE)
    svc.fit(K_train, y_train)
    kernel_pred = svc.predict(K_test).astype(int)
    kernel_prob = svc.predict_proba(K_test)[:, 1]

    # ---- VQC (seed 42, same training as vqc.py) ----
    print("  training VQC...", flush=True)
    weights = train_vqc(pnp.array(Xs_train), y_train)
    weights_np = np.asarray(weights, dtype=float)
    vqc_scores = np.array([float(clean_vqc_circuit(x, weights)) for x in Xs_test])
    vqc_pred = (vqc_scores > 0).astype(int)

    # ---- save per-patient predictions ----
    feature_cols = [f"feature_{i}" for i in range(1, X_test.shape[1] + 1)]
    df = pd.DataFrame(X_test, columns=feature_cols)
    df.insert(0, "true_label", y_test)
    df["kernel_pred"] = kernel_pred
    df["kernel_prob_positive"] = np.round(kernel_prob, 4)
    df["kernel_correct"] = (kernel_pred == y_test).astype(int)
    df["vqc_score"] = np.round(vqc_scores, 4)
    df["vqc_pred"] = vqc_pred
    df["vqc_correct"] = (vqc_pred == y_test).astype(int)
    df.index.name = "patient_id"
    df.to_csv(os.path.join(OUT_DIR, f"{name}_predictions.csv"))

    # ---- save artifacts for inference.py ----
    np.save(os.path.join(OUT_DIR, f"{name}_vqc_weights.npy"), weights_np)
    with open(os.path.join(OUT_DIR, f"{name}_artifacts.pkl"), "wb") as f:
        pickle.dump({
            "dataset": name,
            "scaler": scaler,
            "X_train_scaled": Xs_train,
            "y_train": y_train,
            "svc": svc,
            "vqc_weights": weights_np,
            "feature_names": feature_cols,
        }, f)

    # ---- metrics ----
    result = {
        "dataset": name,
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "kernel_svm": metrics(y_test, kernel_pred),
        "vqc_seed42": metrics(y_test, vqc_pred),
    }
    with open(os.path.join(OUT_DIR, f"{name}_metrics.json"), "w") as f:
        json.dump(result, f, indent=2)

    print(f"  Kernel SVM accuracy: {result['kernel_svm']['accuracy']:.4f} "
          f"| VQC (seed 42) accuracy: {result['vqc_seed42']['accuracy']:.4f}")
    print("  (compare with your benchmark table: these should match it)", flush=True)
    return result


def update_combined(result):
    path = os.path.join(OUT_DIR, "metrics_all.json")
    combined = {}
    if os.path.exists(path):
        with open(path) as f:
            combined = json.load(f)
    combined[result["dataset"]] = result
    with open(path, "w") as f:
        json.dump(combined, f, indent=2)


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else "heart_disease"
    if arg == "all":
        names = VALID
    elif arg in VALID:
        names = [arg]
    else:
        print(f"Unknown dataset '{arg}'. Choose from: {VALID + ['all']}")
        sys.exit(1)
    for n in names:
        update_combined(export_dataset(n))
    print(f"\nDone. Files are in: {OUT_DIR}")
