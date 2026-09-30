"""
make_preprocess.py  -  rebuild the raw -> 6 PCA features pipeline for breast_cancer.

Run from E:\\SIH (same folder as exported_results):
    python make_preprocess.py

It tries the likely ways Person A could have built the pipeline (10 mean
columns vs all 30, several split settings) and ONLY saves
exported_results/breast_cancer_preprocess.pkl when the rebuilt PCA features
match the saved feature_* columns in breast_cancer_predictions.csv.
"""
import os
import pickle
import itertools

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

EXPORT_DIR = "exported_results"
NAME = "breast_cancer"

df = pd.read_csv(os.path.join(EXPORT_DIR, f"{NAME}_predictions.csv"))
saved = df[[c for c in df.columns if c.startswith("feature_")]].values
n_test, n_comp = saved.shape
print(f"Saved test features: {saved.shape}")

Xall, y = load_breast_cancer(return_X_y=True)
column_sets = {"10 means": Xall[:, :10], "all 30": Xall}

for (cols_name, X), test_size, seed, strat in itertools.product(
    column_sets.items(), [0.2, 0.25, 0.3], [42, 0, 1, 7, 123, 2024], [True, False]
):
    Xtr, Xte, _, _ = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y if strat else None
    )
    if len(Xte) != n_test:
        continue
    scaler = StandardScaler().fit(Xtr)
    pca = PCA(n_components=n_comp).fit(scaler.transform(Xtr))
    mine = pca.transform(scaler.transform(Xte))

    # PCA component signs are arbitrary: detect and align them to the saved ones
    signs = np.sign((mine * saved).sum(axis=0))
    signs[signs == 0] = 1
    aligned = mine * signs
    if np.allclose(aligned, saved, atol=1e-3):
        pca.components_ = pca.components_ * signs[:, None]
        out = os.path.join(EXPORT_DIR, f"{NAME}_preprocess.pkl")
        with open(out, "wb") as f:
            pickle.dump({"scaler": scaler, "pca": pca, "n_raw": X.shape[1]}, f)
        print("MATCH FOUND")
        print(f"  columns={cols_name}, test_size={test_size}, random_state={seed}, stratify={strat}")
        print(f"  raw inputs expected: {X.shape[1]}")
        print(f"  saved -> {out}")
        break
else:
    print("NO MATCH. Do not use a guessed pipeline. Ask Person A for their fitted")
    print("scaler + PCA, raw column order, and train_test_split settings.")