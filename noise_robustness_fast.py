"""
Noise Robustness Test - SIH Hybrid QML project
==================================================

WHY THIS MATTERS FOR THE PS:
The problem statement explicitly asks for models "compatible with
near-term quantum hardware," not just a perfect simulator. This script
actually measures that, instead of just claiming it.

WHAT THIS DOES:
1. Trains (or reuses) your VQC and Kernel SVM on clean, noiseless data -
   this represents training on a simulator, which is what everyone does
   today since real quantum hardware access is limited.
2. Then EVALUATES both models through a simulated noisy quantum device
   at increasing noise levels (0%, 1%, 2%, 5%, 10% error rate per gate).
   This represents what would happen if you deployed the same trained
   model onto real, imperfect quantum hardware.
3. Plots/prints how accuracy degrades as noise increases.

THE NOISE MODEL USED: Depolarizing channel - a standard, realistic way
to simulate gate errors. At noise level p, after each gate there's a
probability p that the qubit's state gets randomly scrambled. p=0.01
means roughly a 1% chance of error per gate - in the right ballpark for
today's real NISQ devices (varies by hardware, but this order of
magnitude is realistic).

HOW TO RUN:
    python noise_robustness.py heart_disease
    python noise_robustness.py parkinsons
    python noise_robustness.py all      (slower - trains+tests all 4)
"""

import os
import sys
import numpy as np
import pandas as pd
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
N_FEATURES = 6
N_LAYERS = 2
N_EPOCHS = 60
LEARNING_RATE = 0.05
BATCH_SIZE = 16
RANDOM_STATE = 42
NOISE_LEVELS = [0.0, 0.01, 0.02, 0.05, 0.10]  # VQC: all levels (cheap)
KERNEL_NOISE_LEVELS = [0.0, 0.05, 0.10]         # Kernel SVM: fewer levels (expensive)
KERNEL_SUBSAMPLE = 100  # train rows used for the noisy kernel (None = use all)

n_qubits = N_FEATURES


# -----------------------------------------------------------------------
# Data loading (same as before)
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


def angle_encoding(x):
    for i in range(n_qubits):
        qml.RY(x[i], wires=i)


def trainable_layer(weights, layer_idx):
    for i in range(n_qubits):
        qml.RY(weights[layer_idx, i, 0], wires=i)
        qml.RZ(weights[layer_idx, i, 1], wires=i)
    for i in range(n_qubits):
        qml.CNOT(wires=[i, (i + 1) % n_qubits])


def add_noise(noise_prob):
    """Applies a depolarizing error to every qubit - call this after any gate layer."""
    if noise_prob > 0:
        for i in range(n_qubits):
            qml.DepolarizingChannel(noise_prob, wires=i)


# -----------------------------------------------------------------------
# VQC: clean training (identical to vqc.py), then noisy evaluation
# -----------------------------------------------------------------------
clean_dev = qml.device("default.qubit", wires=n_qubits)


@qml.qnode(clean_dev)
def clean_vqc_circuit(x, weights):
    angle_encoding(x)
    for layer_idx in range(N_LAYERS):
        trainable_layer(weights, layer_idx)
    return qml.expval(qml.PauliZ(0))


def square_loss(labels, predictions):
    labels_pm1 = 2 * labels - 1
    return pnp.mean((labels_pm1 - predictions) ** 2)


def cost(weights, X_batch, y_batch):
    predictions = pnp.stack([clean_vqc_circuit(x, weights) for x in X_batch])
    return square_loss(y_batch, predictions)


def train_vqc(X_train, y_train):
    """Identical training procedure to vqc.py - trains on the clean simulator."""
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


def make_noisy_vqc_circuit(noise_prob):
    """
    Builds a VQC circuit that injects noise after the encoding step and
    after each trainable layer - simulating what happens if this exact
    trained model ran on real, imperfect quantum hardware.
    """
    noisy_dev = qml.device("default.mixed", wires=n_qubits)

    @qml.qnode(noisy_dev)
    def noisy_circuit(x, weights):
        angle_encoding(x)
        add_noise(noise_prob)
        for layer_idx in range(N_LAYERS):
            trainable_layer(weights, layer_idx)
            add_noise(noise_prob)
        return qml.expval(qml.PauliZ(0))

    return noisy_circuit


def evaluate_vqc_under_noise(weights, X_test, y_test, noise_prob):
    noisy_circuit = make_noisy_vqc_circuit(noise_prob)
    y_pred = np.array([1 if noisy_circuit(x, weights) > 0 else 0 for x in X_test])
    return accuracy_score(y_test, y_pred)


# -----------------------------------------------------------------------
# Kernel SVM: recompute kernel matrix under noise, retrain SVM each level
# (this represents running the ENTIRE kernel pipeline on noisy hardware,
# since a real deployment would compute similarities on the noisy device
# too, not just at "training time")
# -----------------------------------------------------------------------
def make_noisy_kernel_circuit(noise_prob):
    noisy_dev = qml.device("default.mixed", wires=n_qubits)

    @qml.qnode(noisy_dev)
    def noisy_kernel_circuit(x1, x2):
        angle_encoding(x1)
        add_noise(noise_prob)
        qml.adjoint(angle_encoding)(x2)
        add_noise(noise_prob)
        return qml.probs(wires=range(n_qubits))

    return noisy_kernel_circuit


def compute_kernel_matrix_noisy(A, B, noise_prob):
    kernel_fn = make_noisy_kernel_circuit(noise_prob)
    return np.array([[kernel_fn(a, b)[0] for b in B] for a in A])


def evaluate_kernel_svm_under_noise(X_train, X_test, y_train, y_test, noise_prob):
    K_train = compute_kernel_matrix_noisy(X_train, X_train, noise_prob)
    K_test = compute_kernel_matrix_noisy(X_test, X_train, noise_prob)
    model = SVC(kernel="precomputed")
    model.fit(K_train, y_train)
    y_pred = model.predict(K_test)
    return accuracy_score(y_test, y_pred)


# -----------------------------------------------------------------------
# Run the full noise sweep
# -----------------------------------------------------------------------
def run(dataset_name):
    print(f"\n{'='*60}\nNoise Robustness Test: {dataset_name}\n{'='*60}")

    X_train, X_test, y_train, y_test = load_dataset(dataset_name)
    X_train, X_test = rescale_for_angles(X_train, X_test)

    print("Training VQC on clean simulator (one-time)...")
    weights = train_vqc(pnp.array(X_train), y_train)

    print(f"\n{'Noise Level':>12} | {'VQC Accuracy':>13} | {'Kernel SVM Accuracy':>20}")
    print("-" * 52)

    # Subsample training rows for the kernel step only (VQC is unaffected).
    # Kernel cost scales with rows^2, so 100 rows is ~5x cheaper than 237.
    if KERNEL_SUBSAMPLE is not None and len(X_train) > KERNEL_SUBSAMPLE:
        rng = np.random.RandomState(RANDOM_STATE)
        idx = rng.choice(len(X_train), KERNEL_SUBSAMPLE, replace=False)
        Xk_train, yk_train = X_train[idx], y_train[idx]
        print(f"(Kernel SVM uses a {KERNEL_SUBSAMPLE}-row subsample - its 0% number "
              f"will differ from the full-data run, compare only within this table.)")
    else:
        Xk_train, yk_train = X_train, y_train

    import time
    results = []
    for noise_prob in NOISE_LEVELS:
        print(f"  [{noise_prob*100:.0f}% noise]", flush=True)
        vqc_acc = evaluate_vqc_under_noise(weights, X_test, y_test, noise_prob)

        kernel_acc = None
        if noise_prob in KERNEL_NOISE_LEVELS:
            t0 = time.time()
            kernel_acc = evaluate_kernel_svm_under_noise(
                Xk_train, X_test, yk_train, y_test, noise_prob
            )
            print(f"    Kernel SVM done in {time.time()-t0:.0f}s", flush=True)

        results.append({"noise": noise_prob, "vqc_acc": vqc_acc, "kernel_acc": kernel_acc})
        k = f"{kernel_acc:.4f}" if kernel_acc is not None else "  skipped"
        print(f"{noise_prob*100:>10.0f}% | VQC {vqc_acc:.4f} | Kernel SVM {k}", flush=True)

    return dataset_name, results


def print_all_summary(all_results):
    print(f"\n{'='*70}\nFULL NOISE ROBUSTNESS SUMMARY\n{'='*70}")
    for dataset_name, results in all_results:
        print(f"\n{dataset_name}:")
        for r in results:
            k = f"{r['kernel_acc']:.4f}" if r['kernel_acc'] is not None else "skipped"
            print(f"  {r['noise']*100:>5.0f}% noise -> VQC: {r['vqc_acc']:.4f} | Kernel SVM: {k}")


if __name__ == "__main__":
    valid = ["breast_cancer", "diabetes", "heart_disease", "parkinsons"]
    dataset_name = sys.argv[1] if len(sys.argv) > 1 else "heart_disease"

    if dataset_name == "all":
        all_results = [run(name) for name in valid]
        print_all_summary(all_results)
    elif dataset_name in valid:
        run(dataset_name)
    else:
        print(f"Unknown dataset '{dataset_name}'. Choose from: {valid + ['all']}")
        sys.exit(1)