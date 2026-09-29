import pandas as pd
df = pd.read_csv("quantum_ready_diabetes_train.csv")
print(df["label"].value_counts())