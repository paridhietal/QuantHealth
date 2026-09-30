import joblib, numpy as np, warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")

def load(dataset, root="."):
    return [joblib.load(f"{root}/{dataset}/{n}.joblib") for n in ("scaler", "pca", "angle_scaler")]

def to_quantum_features(steps, row_values):
    x = np.asarray(row_values, dtype=float).reshape(1, -1)   # raw values, in spec order
    for step in steps:
        x = step.transform(x)
    return x[0]                                              # 6 angles for the quantum encoder

if __name__ == "__main__":
    steps = load("heart_disease")
    mean_patient = steps[0].mean_                            # dummy input just to test the pipe
    print(to_quantum_features(steps, mean_patient))
