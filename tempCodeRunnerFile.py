"""
Multi-seed check: is VQC's heart_disease result real, or noise?
==================================================================

WHY THIS MATTERS:
Your heart_disease test set has only 60 patients. VQC got 86.7% (52/60
correct), classical got 83.3% (50/60 correct) - a 2-patient difference.
On a sample this small, that gap could easily be random luck from how
the VQC's weights happened to initialize, not a real quantum advantage.

WHAT THIS SCRIPT DOES:
Runs the exact same VQC training on heart_disease, 5 times, each with a
different random seed (which changes weight initialization and the
shuffle order of mini-batches during training - NOT the train/test split,
which stays fixed since that came from Person A's files).

HOW TO READ THE RESULT:
- If accuracy stays consistently above 83.3% (classical's score) across
  most/all seeds -> the win is likely real and reproducible, cite it
  confidently.
- If it bounces both above AND below 83.3% depending on the seed -> the
  "win" was likely a lucky roll, not a genuine quantum advantage. Be
  honest about this if asked - say you checked for exactly this and be
  upfront about what you found either way.
"""

import numpy as np
import pandas as pd
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import os

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


def run_seed_check():
    X_train, X_test, y_train, y_test = load_dataset("heart_disease")
    X_train, X_test = rescale_for_angles(X_train, X_test)

    print(f"Running VQC on heart_disease across {len(SEEDS)} seeds...")
    print("(classical baseline for comparison: 83.3% accuracy)\n")

    accuracies = []
    for seed in SEEDS:
        weights = train_vqc(pnp.array(X_train), y_train, seed)
        y_pred = np.array([predict_label(x, weights) for x in X_test])
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        beat_classical = "BEATS classical" if acc > 0.8333 else "does NOT beat classical"
        print(f"  Seed {seed:5d} | Accuracy: {acc:.4f}  ({beat_classical})")

    accuracies = np.array(accuracies)
    print(f"\n{'='*50}")
    print(f"Mean accuracy across seeds: {accuracies.mean():.4f}")
    print(f"Std deviation:              {accuracies.std():.4f}")
    print(f"Range:                      {accuracies.min():.4f} - {accuracies.max():.4f}")
    print(f"Seeds beating classical (83.3%): {(accuracies > 0.8333).sum()} / {len(SEEDS)}")
    print(f"{'='*50}")

    if (accuracies > 0.8333).sum() >= 4:
        print("\n-> Result looks REAL and reproducible. Cite it confidently.")
    elif (accuracies > 0.8333).sum() <= 1:
        print("\n-> Original result looks like it was likely a LUCKY seed, not a real effect.")
        print("   Be honest about this - report the mean instead of the single best run.")
    else:
        print("\n-> Mixed result - VQC is COMPETITIVE with classical here, not clearly better.")
        print("   Honest framing: 'VQC performs comparably to classical on heart_disease,")
        print("   occasionally exceeding it' - still a fine, defensible thing to say.")


if __name__ == "__main__":
    run_seed_check()