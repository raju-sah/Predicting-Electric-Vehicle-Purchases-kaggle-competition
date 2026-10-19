import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

train = pd.read_csv("train.csv")
y = (train["Will_Buy_EV"] == "Yes").astype(int).values

mask_yes = (train["Home_Charging_Possible"] == "Yes")
mask_no = ~mask_yes

print(f"Total train: {len(train):,}")
print(f"Home Charging YES: {mask_yes.sum():,} ({mask_yes.mean():.2%}), Pos rate: {y[mask_yes].mean():.4f}")
print(f"Home Charging NO:  {mask_no.sum():,} ({mask_no.mean():.2%}), Pos rate: {y[mask_no].mean():.4f}")

# Check performance of top OOFs within each subset:
naji_oof = pd.read_csv("scratch/najiama_out/oof_XGBOOST.csv")["OOF_Pred"].values
deotte_oof = pd.read_csv("scratch/deotte_gpu_out/oof_xgb_deotte_simpson.csv")["oof_prediction"].values

for name, oof in [("Najiama XGB", naji_oof), ("Deotte Simpson XGB", deotte_oof)]:
    auc_all = roc_auc_score(y, oof)
    auc_yes = roc_auc_score(y[mask_yes], oof[mask_yes])
    auc_no = roc_auc_score(y[mask_no], oof[mask_no])
    print(f"\n{name}:")
    print(f"  Overall AUC: {auc_all:.6f}")
    print(f"  YES subset AUC: {auc_yes:.6f}")
    print(f"  NO subset AUC:  {auc_no:.6f}")
