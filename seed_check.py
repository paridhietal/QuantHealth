"""
Multi-seed check for VQC (any dataset) - SIH Hybrid QML project
===================================================================

WHY: a VQC's result depends on its random starting weights and shuffle
order. One lucky or unlucky seed can flip a comparison with classical, so
report the MEAN and SPREAD across seeds, not the best run.

WHAT THIS DOES: trains the same VQC once per seed (train/test split fixed,
from Person A's CSVs) and prints each seed's accuracy, how many patients
that is above/below the classical baseline, then mean, std and range.

HOW TO RUN (same folder as the CSVs):
    python -u seed_check.py heart_disease
    python -u seed_check.py parkinsons
    python -u seed_check.py all          (slow: breast_cancer and diabetes are big)
"""

import numpy as np
import pandas as pd
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import os
import sys

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
N_FEATURES = 6
N_LAYERS = 2
N_EPOCHS = 60
LEARNING_RATE = 0.05
BATCH_SIZE = 16
SEEDS = [42, 7, 123, 2024, 99]  # 5 different random seeds to test

n_qubits = N_FEATURES
dev = qml.device("default.qubit", wires=n_qubits)


def load_dataset(name):
    train_df = pd.read_csv(f"{DATA_DIR}/quantum_ready_{name}_train.csv")
    test_df = pd.read_csv(f"{DATA_DIR}/quantum_ready_{name}_test.csv")
    feature_cols = [f"feature_{i}" for i in range(1, N_FEATURES + 1)]
    X_train = train_df[feature_cols].values
    y_train = train_df["label"].values
    X_test = test_df[feature_cols].values
    y_test = test_df["label"].values
    return X_train, X_test, y_train, y_test


def rescale_for_angles(X_train, X_test):
    scaler = MinMaxScaler(feature_range=(-np.pi, np.pi))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


def data_encoding(x):
    for i in range(n_qubits):
        qml.RY(x[i], wires=i)


def trainable_layer(weights, layer_idx):
    for i in range(n_qubits):
        qml.RY(weights[layer_idx, i, 0], wires=i)
        qml.RZ(weights[layer_idx, i, 1], wires=i)
    for i in range(n_qubits):
        qml.CNOT(wires=[i, (i + 1) % n_qubits])


@qml.qnode(dev)
def circuit(x, weights):
    data_encoding(x)
    for layer_idx in range(N_LAYERS):
        trainable_layer(weights, layer_idx)
    return qml.expval(qml.PauliZ(0))


def predict_label(x, weights):
    return 1 if circuit(x, weights) > 0 else 0


def square_loss(labels, predictions):
    labels_pm1 = 2 * labels - 1
    return pnp.mean((labels_pm1 - predictions) ** 2)


def cost(weights, X_batch, y_batch):
    predictions = pnp.stack([circuit(x, weights) for x in X_batch])
    return square_loss(y_batch, predictions)


def train_vqc(X_train, y_train, seed):
    np.random.seed(seed)
    weights = pnp.array(
        np.random.uniform(0, 2 * np.pi, size=(N_LAYERS, n_qubits, 2)),
        requires_grad=True,
    )
    optimizer = qml.AdamOptimizer(stepsize=LEARNING_RATE)
    n_samples = len(X_train)
    best_loss = float("inf")
    best_weights = weights

    for epoch in range(N_EPOCHS):
        perm = np.random.permutation(n_samples)
        X_shuffled = X_train[perm]
        y_shuffled = y_train[perm]
        epoch_loss = 0.0
        n_batches = max(1, n_samples // BATCH_SIZE)
        for b in range(n_batches):
            start = b * BATCH_SIZE
            end = start + BATCH_SIZE
            X_batch = X_shuffled[start:end]
            y_batch = y_shuffled[start:end]
            if len(X_batch) == 0:
                continue
            weights, batch_loss = optimizer.step_and_cost(
                lambda w: cost(w, X_batch, y_batch), weights
            )
            epoch_loss += batch_loss
        avg_epoch_loss = epoch_loss / n_batches
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_weights = pnp.array(weights, requires_grad=False)

    return best_weights


CLASSICAL_BASELINE = {
    "breast_cancer": 0.9649,
    "diabetes": 0.7208,
    "heart_disease": 0.8333,
    "parkinsons": 0.9487,
}


def run_seed_check(dataset_name):
    baseline = CLASSICAL_BASELINE[dataset_name]
    X_train, X_test, y_train, y_test = load_dataset(dataset_name)
    X_train, X_test = rescale_for_angles(X_train, X_test)
    n_test = len(y_test)

    print(f"\nVQC on {dataset_name} across {len(SEEDS)} seeds "
          f"(classical baseline {baseline*100:.1f}%, {n_test} test patients)\n", flush=True)

    accs = []
    for seed in SEEDS:
        weights = train_vqc(pnp.array(X_train), y_train, seed)
        y_pred = np.array([predict_label(x, weights) for x in X_test])
        acc = accuracy_score(y_test, y_pred)
        accs.append(acc)
        gap = round((acc - baseline) * n_test)
        print(f"  Seed {seed:5d} | Accuracy: {acc:.4f}  ({gap:+d} patients vs classical)", flush=True)

    accs = np.array(accs)
    mean_gap = (accs.mean() - baseline) * n_test
    spread = (accs.max() - accs.min()) * n_test

    print(f"\n{'='*56}")
    print(f"Mean accuracy: {accs.mean():.4f}   Std: {accs.std():.4f}")
    print(f"Range: {accs.min():.4f} - {accs.max():.4f}  (spread {spread:.0f} patients)")
    print(f"Mean vs classical: {mean_gap:+.1f} patients")
    print(f"{'='*56}")

    if abs(mean_gap) <= 1:
        verdict = "On average COMPARABLE to classical (within 1 patient)."
    elif mean_gap > 1:
        verdict = "On average ABOVE classical"
        verdict += "; every seed at or above it." if accs.min() >= baseline - 1e-9 else ", but some seeds fall below it."
    else:
        verdict = "On average BELOW classical."
    print(verdict)
    print("Report mean +- std across seeds, not the best single run.")


if __name__ == "__main__":
    valid = list(CLASSICAL_BASELINE)
    name = sys.argv[1] if len(sys.argv) > 1 else "heart_disease"
    if name == "all":
        for d in valid:
            run_seed_check(d)
    elif name in valid:
        run_seed_check(name)
    else:
        print(f"Unknown dataset '{name}'. Choose from: {valid + ['all']}")
        sys.exit(1)