"""
Quantum Kernel SVM - runs on Person A's real quantum-ready datasets
=====================================================================

WHAT THIS DOES (plain English):
1. Loads Person A's pre-split, PCA-reduced CSVs for a chosen disease
2. Rescales the 6 features into [-pi, pi] (angle encoding needs this range;
   Person A's data is standardized (~-3 to 3) but not yet in that range)
3. Encodes each patient into a quantum circuit ("angle encoding")
4. Computes a "quantum kernel" = similarity between every pair of patients,
   using the quantum circuit as the similarity measure
5. Feeds that similarity matrix into a normal SVM classifier
6. Reports accuracy / sensitivity / specificity - same metrics your
   classical baseline (Person A) should be reporting, for a fair comparison

HOW TO RUN:
    python quantum_kernel_svm.py breast_cancer
    python quantum_kernel_svm.py diabetes
    python quantum_kernel_svm.py heart_disease
    python quantum_kernel_svm.py parkinsons

IMPORTANT ON SCALING:
The [-pi, pi] rescaler is fit ONLY on the training set, then applied to
the test set. This avoids "data leakage" (letting test-set information
sneak into training) - a classic mistake that would make your reported
accuracy look better than it really is, and one a sharp judge might ask
about directly. Always be able to say "we fit our scaler on train only."
"""

import sys
import numpy as np
import pandas as pd
import pennylane as qml
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import os
DATA_DIR = os.path.dirname(os.path.abspath(__file__))  # same folder as this script
N_FEATURES = 6  # matches Person A's PCA output -> 6 qubits


# -----------------------------------------------------------------------
# STEP 1: Load Person A's real data
# -----------------------------------------------------------------------
def load_dataset(name):
    """
    name: one of 'breast_cancer', 'diabetes', 'heart_disease', 'parkinsons'
    Expects files named quantum_ready_{name}_train.csv / _test.csv,
    columns feature_1..feature_6 plus a label column.
    """
    train_df = pd.read_csv(f"{DATA_DIR}/quantum_ready_{name}_train.csv")
    test_df = pd.read_csv(f"{DATA_DIR}/quantum_ready_{name}_test.csv")

    feature_cols = [f"feature_{i}" for i in range(1, N_FEATURES + 1)]

    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values

    return X_train, X_test, y_train, y_test


# -----------------------------------------------------------------------
# STEP 2: Rescale into angle-encoding range [-pi, pi], fit on train only
# -----------------------------------------------------------------------
def rescale_for_angles(X_train, X_test):
    scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)  # transform only - no leakage
    return X_train_scaled, X_test_scaled


# -----------------------------------------------------------------------
# STEP 3: Quantum encoding + kernel definition
# -----------------------------------------------------------------------
n_qubits = N_FEATURES
dev = qml.device("default.qubit", wires=n_qubits)


def angle_encoding(x):
    """Each feature -> rotation angle on its own qubit, then entangle."""
    for i in range(n_qubits):
        qml.RY(x[i], wires=i)
    for i in range(n_qubits):
        qml.CNOT(wires=[i, (i + 1) % n_qubits])


@qml.qnode(dev)
def kernel_circuit(x1, x2):
    """Overlap between two encoded points -> similarity score."""
    angle_encoding(x1)
    qml.adjoint(angle_encoding)(x2)
    return qml.probs(wires=range(n_qubits))


def quantum_kernel(x1, x2):
    return kernel_circuit(x1, x2)[0]  # prob of measuring all-zeros


def compute_kernel_matrix(A, B):
    """Pairwise similarity matrix - O(len(A)*len(B)) circuit evaluations."""
    return np.array([[quantum_kernel(a, b) for b in B] for a in A])


# -----------------------------------------------------------------------
# STEP 4: Train + evaluate
# -----------------------------------------------------------------------
def run(dataset_name):
    print(f"\n{'='*60}\nDataset: {dataset_name}\n{'='*60}")

    print("Loading data...")
    X_train, X_test, y_train, y_test = load_dataset(dataset_name)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    print("Rescaling to angle-encoding range [-pi, pi]...")
    X_train, X_test = rescale_for_angles(X_train, X_test)

    print(f"Computing training kernel matrix ({len(X_train)}x{len(X_train)})...")
    K_train = compute_kernel_matrix(X_train, X_train)

    print(f"Computing test kernel matrix ({len(X_test)}x{len(X_train)})...")
    K_test = compute_kernel_matrix(X_test, X_train)

    print("Training SVM on quantum kernel...")
    model = SVC(kernel="precomputed")
    model.fit(K_train, y_train)

    y_pred = model.predict(K_test)

    print(f"\n--- Quantum Kernel SVM Results: {dataset_name} ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    print(f"Sensitivity (catching sick patients): {sensitivity:.4f}")
    print(f"Specificity (correctly clearing healthy): {specificity:.4f}")
    print("\nFull report:")
    print(classification_report(y_test, y_pred))

    return {
        "dataset": dataset_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "model": model,
        "y_test": y_test,
        "y_pred": y_pred,
    }


def print_summary_table(results):
    print(f"\n{'='*60}\nSUMMARY - all datasets\n{'='*60}")
    print(f"{'Dataset':<15} {'Accuracy':>10} {'Sensitivity':>13} {'Specificity':>13}")
    for r in results:
        print(f"{r['dataset']:<15} {r['accuracy']:>10.4f} {r['sensitivity']:>13.4f} {r['specificity']:>13.4f}")


if __name__ == "__main__":
    valid = ["breast_cancer", "diabetes", "heart_disease", "parkinsons"]
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "breast_cancer"

    if dataset_name == "all":
        results = [run(name) for name in valid]
        print_summary_table(results)
    elif dataset_name in valid:
        run(dataset_name)
    else:
        print(f"Unknown dataset '{dataset_name}'. Choose from: {valid + ['all']}")
        sys.exit(1)