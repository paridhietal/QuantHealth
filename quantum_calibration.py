"""
Calibration + triage for the quantum models - SIH Hybrid QML project
========================================================================

GOAL:
Turn each model's output into a probability ("82% risk") and check whether
that probability can be trusted, then show what a confident / uncertain
triage would look like. Teammate's idea, extended to the quantum models.

WHAT THIS DOES (one dataset at a time):
  1. VQC: its raw output is a score in [-1, 1]. We map it to a probability
     with Platt scaling (a 1-feature logistic regression). To avoid
     calibrating on the same patients the model trained on, the VQC here is
     trained on 75% of the training rows and the other 25% is used to fit
     the calibration. So its accuracy will differ a little from the
     full-data VQC number (86.7% on heart_disease). That is expected.
  2. Kernel SVM: scikit-learn's SVC(probability=True) does its own internal
     Platt scaling. Trained on all training rows, clean (noiseless) kernel.
  3. Reference: plain Logistic Regression on the same features, so you can
     compare "whose confidence is more trustworthy". This is a quick
     reference, not necessarily Person A's exact model.
  4. For each model it reports:
       - Accuracy at a 0.5 cutoff
       - Brier score (mean squared error of the probability; LOWER is
         better, and 0.25 is what you get by always saying 50%)
       - Triage: "confident" if p >= 0.85 or p <= 0.15, otherwise
         "uncertain". Reports how many patients are confident and the
         accuracy on each group.
       - A reliability table (predicted probability vs how often the
         patient was actually positive)
  5. Saves a reliability diagram as calibration_<dataset>.png (needs matplotlib)

HOW TO RUN (same folder as noise_robustness.py, quantum_kernel_svm.py, CSVs):
    python quantum_calibration.py heart_disease
    python quantum_calibration.py parkinsons

RUNTIME: the clean kernel matrix takes a few minutes on heart_disease, plus
about a minute for VQC training. Run it in a normal terminal.

HONEST CAVEAT: with 39-60 test patients, each reliability bin holds only a
handful of patients. Treat the diagram as illustrative and rely on the Brier
score and triage counts, and do not over-read small differences.
"""

import sys
import numpy as np
from pennylane import numpy as pnp
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, brier_score_loss

RANDOM_STATE = 42
HI, LO = 0.85, 0.15   # triage thresholds
N_BINS = 5


# -----------------------------------------------------------------------
# Reporting helpers (no quantum code in here)
# -----------------------------------------------------------------------
def reliability_table(y_true, p, n_bins=N_BINS):
    """Returns rows of (bin_low, bin_high, n, mean_predicted, frac_positive)."""
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            rows.append((edges[b], edges[b + 1], 0, float("nan"), float("nan")))
        else:
            rows.append((edges[b], edges[b + 1], n, float(p[mask].mean()), float(y_true[mask].mean())))
    return rows


def safe_acc(y_true, pred, mask):
    if mask.sum() == 0:
        return float("nan")
    return accuracy_score(y_true[mask], pred[mask])


def report(name, y_true, p):
    pred = (p > 0.5).astype(int)
    acc = accuracy_score(y_true, pred)
    brier = brier_score_loss(y_true, p)
    conf = (p >= HI) | (p <= LO)
    unc = ~conf
    out = {
        "name": name,
        "acc": acc,
        "brier": brier,
        "n_conf": int(conf.sum()),
        "acc_conf": safe_acc(y_true, pred, conf),
        "n_unc": int(unc.sum()),
        "acc_unc": safe_acc(y_true, pred, unc),
        "table": reliability_table(y_true, p),
        "p": p,
    }
    n = len(y_true)
    print(f"\n--- {name} ---")
    print(f"Accuracy (cutoff 0.5): {acc:.4f} | Brier score: {brier:.4f}")
    print(f"Triage (confident = p>={HI} or p<={LO}):")
    print(f"  confident: {out['n_conf']}/{n} patients, accuracy {out['acc_conf']:.4f}")
    print(f"  uncertain: {out['n_unc']}/{n} patients, accuracy {out['acc_unc']:.4f}")
    print(f"Reliability ({N_BINS} bins):  range | n | mean predicted | fraction actually positive")
    for lo, hi, cnt, mp, fp in out["table"]:
        mp_s = f"{mp:.2f}" if cnt else "  - "
        fp_s = f"{fp:.2f}" if cnt else "  - "
        print(f"  {lo:.1f}-{hi:.1f} | {cnt:>2} | {mp_s} | {fp_s}")
    return out


def save_reliability_plot(results, dataset_name):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n(matplotlib not installed - skipping the diagram. pip install matplotlib)")
        return
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="perfectly calibrated")
    for r in results:
        pts = [(mp, fp) for _, _, cnt, mp, fp in r["table"] if cnt > 0]
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="o", linewidth=2, label=f"{r['name']} (Brier {r['brier']:.3f})")
    ax.set_xlabel("Predicted probability of disease")
    ax.set_ylabel("Fraction actually positive")
    ax.set_title(f"Reliability diagram: {dataset_name}\n(few patients per bin - illustrative)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = f"calibration_{dataset_name}.png"
    fig.savefig(path, dpi=200)
    print(f"\nSaved {path}")


# -----------------------------------------------------------------------
# Model probability builders
# -----------------------------------------------------------------------
def vqc_probabilities(X_train, y_train, X_test):
    from noise_robustness import train_vqc, clean_vqc_circuit

    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_train, y_train, test_size=0.25, random_state=RANDOM_STATE, stratify=y_train
    )
    print(f"Training VQC on {len(X_fit)} rows, calibrating on {len(X_cal)} held-out rows...", flush=True)
    weights = train_vqc(pnp.array(X_fit), y_fit)

    out_cal = np.array([float(clean_vqc_circuit(x, weights)) for x in X_cal]).reshape(-1, 1)
    out_test = np.array([float(clean_vqc_circuit(x, weights)) for x in X_test]).reshape(-1, 1)

    platt = LogisticRegression(C=100.0).fit(out_cal, y_cal)
    p_platt = platt.predict_proba(out_test)[:, 1]
    p_raw = (out_test.ravel() + 1.0) / 2.0   # naive mapping of [-1,1] to [0,1]
    return p_raw, p_platt


def kernel_probabilities(X_train, y_train, X_test):
    from quantum_kernel_svm import compute_kernel_matrix

    print(f"Computing clean quantum kernel matrices ({len(X_train)}x{len(X_train)}) - takes a few minutes...", flush=True)
    K_train = compute_kernel_matrix(X_train, X_train)
    K_test = compute_kernel_matrix(X_test, X_train)
    svc = SVC(kernel="precomputed", probability=True, random_state=RANDOM_STATE)
    svc.fit(K_train, y_train)
    return svc.predict_proba(K_test)[:, 1]


def classical_probabilities(X_train, y_train, X_test):
    lr = LogisticRegression(max_iter=1000).fit(X_train, y_train)
    return lr.predict_proba(X_test)[:, 1]


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def run(dataset_name):
    from noise_robustness import load_dataset, rescale_for_angles

    print(f"\n{'='*70}\nCalibration + triage: {dataset_name}\n{'='*70}")
    X_train, X_test, y_train, y_test = load_dataset(dataset_name)
    X_train, X_test = rescale_for_angles(X_train, X_test)
    y_train = np.asarray(y_train).astype(int)
    y_test = np.asarray(y_test).astype(int)

    base = y_test.mean()
    print(f"Test patients: {len(y_test)}, positives: {int(y_test.sum())}")
    print(f"Reference Brier for always predicting the base rate: {base*(1-base):.4f}")

    p_lr = classical_probabilities(X_train, y_train, X_test)
    p_kernel = kernel_probabilities(X_train, y_train, X_test)
    p_vqc_raw, p_vqc_platt = vqc_probabilities(X_train, y_train, X_test)

    results = [
        report("Classical reference (Logistic Regression)", y_test, p_lr),
        report("Kernel SVM (Platt via SVC)", y_test, p_kernel),
        report("VQC, raw score mapped to [0,1] (uncalibrated)", y_test, p_vqc_raw),
        report("VQC, Platt-calibrated", y_test, p_vqc_platt),
    ]

    print(f"\n{'='*70}\nSUMMARY: {dataset_name}\n{'='*70}")
    print(f"{'Model':<48} {'Acc':>6} {'Brier':>7} {'Confident':>10} {'Acc(conf)':>10}")
    for r in results:
        print(f"{r['name']:<48} {r['acc']:>6.3f} {r['brier']:>7.4f} {r['n_conf']:>5}/{len(y_test):<4} {r['acc_conf']:>10.3f}")

    save_reliability_plot([results[0], results[1], results[3]], dataset_name)


if __name__ == "__main__":
    valid = ["breast_cancer", "diabetes", "heart_disease", "parkinsons"]
    name = sys.argv[1] if len(sys.argv) > 1 else "heart_disease"
    if name not in valid:
        print(f"Unknown dataset '{name}'. Choose from: {valid}")
        sys.exit(1)
    run(name)