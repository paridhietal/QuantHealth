import sys
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.metrics import accuracy_score

# Reuse the exact same data loading / training code as the first noise test
from noise_robustness import (
    load_dataset,
    rescale_for_angles,
    train_vqc,
    angle_encoding,
    trainable_layer,
    add_noise,
    n_qubits,
    N_LAYERS,
)

STRESS_LEVELS = [0.0, 0.05, 0.10, 0.20, 0.30, 0.50]
SHOT_SETTINGS = [1000, 100]
N_REPEATS = 3  # shot-based results are random, so average a few repeats


def make_circuit(noise_prob, shots):
    dev = qml.device("default.mixed", wires=n_qubits, shots=shots)

    @qml.qnode(dev)
    def circuit(x, weights):
        angle_encoding(x)
        add_noise(noise_prob)
        for layer_idx in range(N_LAYERS):
            trainable_layer(weights, layer_idx)
            add_noise(noise_prob)
        return qml.expval(qml.PauliZ(0))

    return circuit


def outputs_and_accuracy(weights, X_test, y_test, noise_prob, shots):
    circuit = make_circuit(noise_prob, shots)
    outs = np.array([float(circuit(x, weights)) for x in X_test])
    y_pred = (outs > 0).astype(int)
    return accuracy_score(y_test, y_pred), np.mean(np.abs(outs))


def run(dataset_name):
    print(f"\n{'='*70}\nVQC noise stress test: {dataset_name}\n{'='*70}")
    X_train, X_test, y_train, y_test = load_dataset(dataset_name)
    X_train, X_test = rescale_for_angles(X_train, X_test)

    print("Training VQC on clean simulator (one-time)...", flush=True)
    weights = train_vqc(pnp.array(X_train), y_train)

    header = f"{'Noise':>6} | {'Exact acc':>9} | {'mean|out|':>9}"
    for s in SHOT_SETTINGS:
        header += f" | {str(s) + ' shots (mean+-std)':>24}"
    print("\n" + header)
    print("-" * len(header))

    for p in STRESS_LEVELS:
        exact_acc, mean_abs = outputs_and_accuracy(weights, X_test, y_test, p, None)
        row = f"{p*100:>5.0f}% | {exact_acc:>9.4f} | {mean_abs:>9.4f}"
        for s in SHOT_SETTINGS:
            accs = [outputs_and_accuracy(weights, X_test, y_test, p, s)[0]
                    for _ in range(N_REPEATS)]
            row += f" | {np.mean(accs):>11.4f} +- {np.std(accs):.4f}    "
        print(row, flush=True)


if __name__ == "__main__":
    valid = ["breast_cancer", "diabetes", "heart_disease", "parkinsons"]
    name = sys.argv[1] if len(sys.argv) > 1 else "heart_disease"
    if name not in valid:
        print(f"Unknown dataset '{name}'. Choose from: {valid}")
        sys.exit(1)
    run(name)