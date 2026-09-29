import pandas as pd
import numpy as np

datasets = ["breast_cancer", "diabetes", "heart_disease", "parkinsons"]
feature_cols = [f"feature_{i}" for i in range(1, 7)]

for name in datasets:
    df = pd.read_csv(f"quantum_ready_{name}_train.csv")
    corr = df[feature_cols].corr().values
    n = corr.shape[0]
    off_diag = corr[~np.eye(n, dtype=bool)]
    avg_abs_corr = np.mean(np.abs(off_diag))
    print(f"{name}: avg |feature correlation| = {avg_abs_corr:.4f}")