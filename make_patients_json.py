import pandas as pd, json, glob

out = {}
for f in glob.glob("exported_results/*_predictions.csv"):
    name = f.split("\\")[-1].replace("_predictions.csv", "")
    out[name] = pd.read_csv(f).to_dict(orient="records")

json.dump(out, open("patients.json", "w"))
print("wrote patients.json with:", list(out.keys()))