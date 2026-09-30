# Q-MedDetect preprocessing (all fitted with scikit-learn 1.6.1)

For every dataset the chain is:  StandardScaler -> PCA(6) -> MinMaxScaler(-pi, pi)
Load `scaler.joblib`, `pca.joblib`, `angle_scaler.joblib` from each folder and apply in that order.
Pin `scikit-learn==1.6.1` (and a compatible numpy) in the dashboard. Also `joblib`.

## Split settings (identical for all datasets; used only for training, not needed at predict time)
train_test_split(test_size=0.2, random_state=42, stratify=y);  PCA(n_components=6, random_state=42)

## Input columns (send in exactly this order; each folder's spec.json has the same list)
### breast_cancer (30, no cleaning)
mean radius, mean texture, mean perimeter, mean area, mean smoothness, mean compactness, mean concavity,
mean concave points, mean symmetry, mean fractal dimension, radius error, texture error, perimeter error,
area error, smoothness error, compactness error, concavity error, concave points error, symmetry error,
fractal dimension error, worst radius, worst texture, worst perimeter, worst area, worst smoothness,
worst compactness, worst concavity, worst concave points, worst symmetry, worst fractal dimension
PCA variance retained: 89.1%

### heart_disease (13)
age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
- NO categorical encoding: sex, cp, fbs, restecg, exang, slope, ca, thal are the raw UCI numeric codes,
  standardized like the continuous columns. Do not one-hot encode.
- Training rows with any missing value were dropped (missing ca/thal); no imputer exists,
  so the dashboard must require every field. Target was binarized (>0 -> 1).
PCA variance retained: 68.8%

### parkinsons (22, no cleaning)
Order (position matters): MDVP:Fo(Hz), MDVP:Fhi(Hz), MDVP:Flo(Hz), MDVP:Jitter(%), MDVP:Jitter(Abs), MDVP:RAP,
MDVP:PPQ, Jitter:DDP, MDVP:Shimmer, MDVP:Shimmer(dB), Shimmer:APQ3, Shimmer:APQ5, MDVP:APQ, Shimmer:DDA,
NHR, HNR, RPDE, DFA, spread1, spread2, D2, PPE
- The saved spec.json / scaler show "MDVP:Jitter" twice and "MDVP:Shimmer" twice (the unit suffixes were
  dropped in the export). Positions 4-5 are Jitter(%) and Jitter(Abs); 9-10 are Shimmer and Shimmer(dB).
- Because of the duplicate stored names, pass a plain numpy array in the order above (not a DataFrame).
PCA variance retained: 92.0%

## Things to know
- Angle scaler was fit on training data, so new patients can land outside [-pi, pi] (test data reached about
  -3.2 .. 4.3). Decide whether to clip; the training data never was.
- The saved scalers are the ones that produced the shipped quantum_ready_*.csv files.
