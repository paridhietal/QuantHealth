"""
Variational Quantum Classifier (VQC) - SIH Hybrid QML project
=================================================================

WHAT THIS DOES DIFFERENTLY FROM THE KERNEL SVM:
The kernel SVM used a FIXED quantum circuit just to measure similarity
between patients, then let a classical SVM do the actual learning.

This VQC is different: the quantum circuit itself IS the model being
trained. It has "trainable weights" (extra rotation angles, separate
from the ones that encode the patient's data) that get adjusted step by
step to make the circuit's output match the correct label more often.
This is conceptually identical to training a normal neural network -
same idea (guess, measure error, adjust, repeat), different substrate.

PIPELINE:
1. Load Person A's pre-split, pre-scaled CSVs (same as kernel script)
2. Rescale to [-pi, pi] for angle encoding (fit on train only)
3. Build a circuit: encode data -> trainable rotation layers -> measure
4. Train with gradient descent (PennyLane computes quantum gradients
   automatically - you don't need to derive any calculus by hand)
5. Evaluate with the same metrics as your kernel SVM and classical
   baseline, for a fair 3-way comparison

HOW TO RUN:
    python vqc.py breast_cancer
    python vqc.py diabetes
    python vqc.py heart_disease
    python vqc.py parkinsons
    python vqc.py all
"""

import os
import sys
import numpy as np
import pandas as pd
import pennylane as qml
from pennylane import numpy as pnp  # PennyLane's autograd-aware numpy
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
N_FEATURES = 6          # matches Person A's PCA output -> 6 qubits
N_LAYERS = 2             # how many trainable rotation+entangle layers to stack
N_EPOCHS = 60            # training iterations - raise if not converging
LEARNING_RATE = 0.05
BATCH_SIZE = 16          # mini-batch size for training steps
RANDOM_STATE = 42


# -----------------------------------------------------------------------
# STEP 1: Load data (same as kernel script)
# -----------------------------------------------------------------------
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


# -----------------------------------------------------------------------
# STEP 2: The variational circuit itself
# -----------------------------------------------------------------------
n_qubits = N_FEATURES
dev = qml.device("default.qubit", wires=n_qubits)


def data_encoding(x):
    """Same angle encoding as the kernel script - encodes the PATIENT'S data."""
    for i in range(n_qubits):
        qml.RY(x[i], wires=i)


def trainable_layer(weights, layer_idx):
    """
    One layer of TRAINABLE rotations + entanglement. These weights are
    NOT the patient's data - they're the model's learned parameters,
    adjusted during training. This is the "neural network" part.
    """
    for i in range(n_qubits):
        qml.RY(weights[layer_idx, i, 0], wires=i)
        qml.RZ(weights[layer_idx, i, 1], wires=i)
    for i in range(n_qubits):
        qml.CNOT(wires=[i, (i + 1) % n_qubits])


@qml.qnode(dev)
def circuit(x, weights):
    """
    Full circuit: encode the patient's data, then run it through
    N_LAYERS of trainable transformations, then measure.
    Output: expectation value in [-1, 1] on qubit 0 - our "prediction score."
    """
    data_encoding(x)
    for layer_idx in range(N_LAYERS):
        trainable_layer(weights, layer_idx)
    return qml.expval(qml.PauliZ(0))


def predict_score(x, weights):
    """Raw circuit output in [-1, 1]."""
    return circuit(x, weights)


def predict_label(x, weights):
    """Convert raw score to a 0/1 label prediction."""
    return 1 if predict_score(x, weights) > 0 else 0


# -----------------------------------------------------------------------
# STEP 3: Training loop
# -----------------------------------------------------------------------
def square_loss(labels, predictions):
    """Mean squared error between true labels (mapped to -1/1) and predictions."""
    labels_pm1 = 2 * labels - 1  # convert 0/1 labels to -1/1 to match circuit output
    return pnp.mean((labels_pm1 - predictions) ** 2)


def cost(weights, X_batch, y_batch):
    predictions = pnp.stack([circuit(x, weights) for x in X_batch])
    return square_loss(y_batch, predictions)


def train_vqc(X_train, y_train):
    np.random.seed(RANDOM_STATE)
    weights = pnp.array(
        np.random.uniform(0, 2 * np.pi, size=(N_LAYERS, n_qubits, 2)),
        requires_grad=True,
    )

    optimizer = qml.AdamOptimizer(stepsize=LEARNING_RATE)
    n_samples = len(X_train)

    best_loss = float("inf")
    best_weights = weights

    for epoch in range(N_EPOCHS):
        # shuffle and take mini-batches each epoch
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

        # keep a copy of the weights from whichever epoch had the lowest
        # loss - avoids reporting a worse model just because training got
        # a bit unstable near the end
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_weights = pnp.array(weights, requires_grad=False)

        if epoch % 10 == 0 or epoch == N_EPOCHS - 1:
            marker = "  <- best so far" if avg_epoch_loss == best_loss else ""
            print(f"  Epoch {epoch:3d} | loss: {avg_epoch_loss:.4f}{marker}")

    print(f"  Using weights from best epoch (loss: {best_loss:.4f})")
    return best_weights


# -----------------------------------------------------------------------
# STEP 4: Evaluate
# -----------------------------------------------------------------------
def evaluate(weights, X_test, y_test):
    y_pred = np.array([predict_label(x, weights) for x in X_test])

    accuracy = accuracy_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Sensitivity (catching sick patients): {sensitivity:.4f}")
    print(f"Specificity (correctly clearing healthy): {specificity:.4f}")
    print("\nFull report:")
    print(classification_report(y_test, y_pred))

    return accuracy, sensitivity, specificity, y_pred


# -----------------------------------------------------------------------
# STEP 5: Run
# -----------------------------------------------------------------------
def run(dataset_name):
    print(f"\n{'='*60}\nDataset: {dataset_name}\n{'='*60}")

    X_train, X_test, y_train, y_test = load_dataset(dataset_name)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")

    X_train, X_test = rescale_for_angles(X_train, X_test)

    print("Training VQC...")
    weights = train_vqc(pnp.array(X_train), y_train)

    print(f"\n--- VQC Results: {dataset_name} ---")
    accuracy, sensitivity, specificity, y_pred = evaluate(weights, X_test, y_test)

    return {
        "dataset": dataset_name,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def print_summary_table(results):
    print(f"\n{'='*60}\nSUMMARY - all datasets (VQC)\n{'='*60}")
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