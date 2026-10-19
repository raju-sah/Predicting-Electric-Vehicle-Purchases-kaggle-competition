import os, glob
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

train = pd.read_csv("train.csv")
y = (train["Will_Buy_EV"] == "Yes").astype(int).values

print("Auditing all potential OOF files with len == 668665...")
found = {}

# Check all npy files
for path in glob.glob("**/*.npy", recursive=True):
    try:
        arr = np.load(path)
        if arr.ndim == 1 and len(arr) == len(y):
            auc = roc_auc_score(y, arr)
            found[path] = (auc, "npy")
    except Exception:
        pass

# Check all csv files
for path in glob.glob("**/*.csv", recursive=True):
    if "train" in path or "test" in path or "sample" in path:
        continue
    try:
        df = pd.read_csv(path, nrows=5)
        cols = [c for c in df.columns if c.lower() not in ["id", "will_buy_ev", "is_train"]]
        if len(cols) == 1:
            full_df = pd.read_csv(path)
            if len(full_df) == len(y):
                arr = full_df[cols[0]].values
                auc = roc_auc_score(y, arr)
                found[path] = (auc, f"csv:{cols[0]}")
    except Exception:
        pass

# Sort by AUC descending
sorted_found = sorted(found.items(), key=lambda x: x[1][0], reverse=True)
print(f"\nFound {len(sorted_found)} candidate OOF arrays:")
for path, (auc, ftype) in sorted_found[:25]:
    print(f"  {path:50s} [{ftype:15s}] : AUC = {auc:.6f}")
