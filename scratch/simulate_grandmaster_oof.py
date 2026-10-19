import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

train = pd.read_csv("train.csv")
y = (train["Will_Buy_EV"] == "Yes").astype(int).values

# Load our top OOFs
naji_oof = pd.read_csv("scratch/najiama_out/oof_XGBOOST.csv")["OOF_Pred"].values
deotte_oof = pd.read_csv("scratch/deotte_gpu_out/oof_xgb_deotte_simpson.csv")["oof_prediction"].values
my_oof = pd.read_csv("scratch/my_gpu_out/oof_xgb_fixed_digits.csv")["oof_prediction"].values
naji_lgb_oof = pd.read_csv("scratch/outputs/najiama_pure-lgbm-model-cv-0-94606-lb-0-94637/oof_LIGHTGBM.csv")["OOF_Pred"].values

# Convert each to rank [0, 1]
rk_naji = rankdata(naji_oof) / len(y)
rk_deotte = rankdata(deotte_oof) / len(y)
rk_my = rankdata(my_oof) / len(y)
rk_lgb = rankdata(naji_lgb_oof) / len(y)

# 4-way tree blend OOF
tree_blend_oof = 0.4312 * rk_naji + 0.2792 * rk_deotte + 0.1660 * rk_my + 0.1236 * rk_lgb
auc_trees = roc_auc_score(y, tree_blend_oof)
print(f"4-way Tree Blend OOF AUC: {auc_trees:.6f}")

# Check zero cells mask on train
zero_cell_tuples = [
    (1.0, "No", "High"),
    (1.0, "Yes", "High"),
    (2.0, "No", "High"),
    (2.0, "No", "Medium"),
    (2.0, "Yes", "High"),
    (3.0, "No", "High"),
    (3.0, "Yes", "High"),
    (4.0, "No", "High"),
    (4.0, "Yes", "High"),
    (5.0, "No", "High"),
]

is_train_zero = pd.Series(False, index=train.index)
for c, s, a in zero_cell_tuples:
    m = (train["Environmental_Concern_Level"] == c) & (train["Subsidy_Available"] == s) & (train["Range_Anxiety_Level"] == a)
    is_train_zero |= m

mask_train_millionaire = (train["Annual_Income_USD"] >= 170537)

# Apply boundary adjustments to tree_blend_oof
adj_oof = tree_blend_oof.copy()
adj_oof[is_train_zero] = adj_oof[is_train_zero] * 0.05
adj_oof[mask_train_millionaire] = adj_oof[mask_train_millionaire] + 10.0

auc_adj = roc_auc_score(y, adj_oof)
print(f"Tree Blend + Boundary Calibration OOF AUC: {auc_adj:.6f} (gain: {auc_adj - auc_trees:+.7f})")
print(f"Number of ties before ordinal ranking: {len(adj_oof) - len(np.unique(adj_oof))}")
rk_adj = rankdata(adj_oof, method="ordinal") / len(adj_oof)
auc_final = roc_auc_score(y, rk_adj)
print(f"Final Ordinal OOF AUC: {auc_final:.6f}")
