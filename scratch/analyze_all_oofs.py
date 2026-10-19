import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata, spearmanr
import glob

# Load train labels
train_path = "train.csv"
train = pd.read_csv(train_path)
y = (train["Will_Buy_EV"] == "Yes").astype(int).values

# Load folds
folds_path = "scratch/golem_lib/folds_seed42.npy"
folds = np.load(folds_path)
print(f"Loaded train: {len(y)} rows, positive rate: {y.mean():.4f}")
print(f"Folds shape: {folds.shape}, unique folds: {np.unique(folds)}")

# Load all Golem OOFs
golem_files = sorted(glob.glob("scratch/golem_lib/oof_*.npy"))
oof_dict = {}
for gf in golem_files:
    name = os.path.basename(gf).replace(".npy", "")
    arr = np.load(gf)
    auc = roc_auc_score(y, arr)
    oof_dict[name] = arr
    print(f"Golem {name:20s}: OOF AUC = {auc:.6f}")

# Check Najiama OOF
najiama_oof_path = "scratch/najiama_out/oof_XGBOOST.csv"
if os.path.exists(najiama_oof_path):
    naji_df = pd.read_csv(najiama_oof_path)
    naji_oof = naji_df["OOF_Pred"].values
    naji_auc = roc_auc_score(y, naji_oof)
    oof_dict["najiama_xgb"] = naji_oof
    print(f"{'najiama_xgb':20s}: OOF AUC = {naji_auc:.6f}")

# Check any other local oof files in scratch or current dir
for extra in glob.glob("scratch/*oof*.npy") + glob.glob("*oof*.npy"):
    try:
        arr = np.load(extra)
        if len(arr) == len(y):
            name = os.path.basename(extra).replace(".npy", "")
            auc = roc_auc_score(y, arr)
            oof_dict[name] = arr
            print(f"Extra {name:20s}: OOF AUC = {auc:.6f}")
    except Exception as e:
        pass

print(f"\nTotal OOF candidate models loaded: {len(oof_dict)}")
