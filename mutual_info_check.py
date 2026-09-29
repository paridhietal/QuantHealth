import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from itertools import combinations

datasets = ["breast_cancer", "diabetes", "heart_disease", "parkinsons"]
feature_cols = [f"feature_{i}" for i in range(1, 7)]

for name in datasets:
    df = pd.read_csv(f"quantum_ready_{name}_train.csv")
    X = df[feature_cols].values
    mi_scores = []
    for i, j in combinations(range(6), 2):
        mi = mutual_info_regression(X[:, [i]], X[:, j], random_state=42)[0]
        mi_scores.append(mi)
    print(f"{name}: avg pairwise mutual information = {np.mean(mi_scores):.4f}")