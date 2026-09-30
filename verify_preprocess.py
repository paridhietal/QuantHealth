"""
verify_preprocess.py  -  run from E:\\SIH

    python verify_preprocess.py

What it does (breast_cancer, fully automatic - uses scikit-learn's built-in data):
  1. Sanity-checks that each preprocess/<disease>/ folder holds the right files
     (scaler input sizes must be 30 / 13 / 22 for breast_cancer / heart_disease / parkinsons).
  2. Rebuilds the test split exactly as in Person A's spec.json
     (test_size=0.2, random_state=42, stratify=y) and runs
        raw -> scaler -> pca            (call this "pca")
        raw -> scaler -> pca -> angle   (call this "angle")
     then compares both with the feature_* columns saved in
     exported_results/breast_cancer_predictions.csv.
     Whichever one matches tells us whether the saved features were ALREADY
     angle-scaled, i.e. whether the live-prediction chain must apply Person A's
     angle_scaler before our own models.
  3. Checks the label coding (scikit-learn: 0 = malignant, 1 = benign).
  4. Writes preprocess/<disease>/mode.json = {"apply_angle_scaler": true/false}
     for every disease folder that exists (same pipeline per spec.json).

Optional, to verify another disease on its own data:
    python verify_preprocess.py heart_disease path\\to\\heart.csv target
    python verify_preprocess.py parkinsons   path\\to\\parkinsons.csv status
"""
import os
import sys
import json
import warnings

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

ROOT = os.path.dirname(os.path.abspath(__file__))
PRE = os.path.join(ROOT, "preprocess")
EXP = os.path.join(ROOT, "exported_results")
EXPECTED_INPUTS = {"breast_cancer": 30, "heart_disease": 13, "parkinsons": 22}


def load_pre(name):
    folder = os.path.join(PRE, name)
    out = {}
    for key in ("scaler", "pca", "angle_scaler"):
        p = os.path.join(folder, f"{key}.joblib")
        if not os.path.exists(p):
            return None, f"missing {p}"
        out[key] = joblib.load(p)
    return out, None


def sanity_check_folders():
    print("== Folder check ==")
    for name, n in EXPECTED_INPUTS.items():
        pre, err = load_pre(name)
        if pre is None:
            print(f"  {name:14s} {err}")
            continue
        got = pre["scaler"].n_features_in_
        ok = "OK" if got == n else f"WRONG FOLDER? expected {n} inputs"
        print(f"  {name:14s} scaler expects {got} inputs -> {ok}")
    print()


def run_check(name, X, y):
    pre, err = load_pre(name)
    if pre is None:
        print(f"[{name}] {err}")
        return None
    if X.shape[1] != pre["scaler"].n_features_in_:
        print(f"[{name}] data has {X.shape[1]} columns but scaler expects {pre['scaler'].n_features_in_}")
        return None

    _, Xte, _, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    Z = pre["pca"].transform(pre["scaler"].transform(Xte))
    A = pre["angle_scaler"].transform(Z)

    csv = os.path.join(EXP, f"{name}_predictions.csv")
    if not os.path.exists(csv):
        print(f"[{name}] {csv} not found")
        return None
    df = pd.read_csv(csv)
    saved = df[[c for c in df.columns if c.startswith("feature_")]].values
    if saved.shape != Z.shape:
        print(f"[{name}] saved features {saved.shape} vs rebuilt {Z.shape}: test split differs")
        return None

    m_pca = bool(np.allclose(Z, saved, atol=1e-3))
    m_angle = bool(np.allclose(A, saved, atol=1e-3))
    tl = df["true_label"].values
    labels = "same as data" if np.array_equal(tl, yte) else ("flipped" if np.array_equal(tl, 1 - yte) else "MISMATCH")

    print(f"[{name}] saved features match scaler->pca only : {m_pca}")
    print(f"[{name}] saved features match ...->angle scaler : {m_angle}")
    print(f"[{name}] saved true_label vs data labels        : {labels}")
    if m_angle:
        return True
    if m_pca:
        return False
    print(f"[{name}] NEITHER matched. Do not trust live predictions; ask Person A.")
    return None


def main():
    sanity_check_folders()

    if len(sys.argv) >= 4:
        name, csv_path, label_col = sys.argv[1], sys.argv[2], sys.argv[3]
        df = pd.read_csv(csv_path)
        y = df[label_col].values
        spec_path = os.path.join(PRE, name, "spec.json")
        cols = None
        if os.path.exists(spec_path):
            cols = json.load(open(spec_path))["input_columns_in_order"]
        if cols and len(set(cols)) == len(cols) and all(c in df.columns for c in cols):
            X = df[cols].values
        else:
            X = df.drop(columns=[label_col]).select_dtypes(include="number").values
        res = run_check(name, X, y)
        print("Result for", name, "->", {True: "apply angle scaler", False: "do NOT apply angle scaler", None: "unverified"}[res])
        return

    from sklearn.datasets import load_breast_cancer
    X, y = load_breast_cancer(return_X_y=True)
    res = run_check("breast_cancer", X, y)
    if res is None:
        print("\nCould not determine the mode, so mode.json was NOT written.")
        return

    for name in EXPECTED_INPUTS:
        folder = os.path.join(PRE, name)
        if os.path.isdir(folder):
            with open(os.path.join(folder, "mode.json"), "w") as f:
                json.dump({"apply_angle_scaler": res, "verified_on": "breast_cancer"}, f)
            print(f"wrote {os.path.join(folder, 'mode.json')}  apply_angle_scaler={res}")
    print("\nRestart uvicorn (Ctrl+C, then run it again) so the backend picks this up.")


if __name__ == "__main__":
    main()