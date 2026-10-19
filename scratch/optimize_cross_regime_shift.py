import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.special import logit, expit

train = pd.read_csv("train.csv")
y = (train["Will_Buy_EV"] == "Yes").astype(int).values

mask_yes = (train["Home_Charging_Possible"] == "Yes").values
mask_no = ~mask_yes

naji_oof = pd.read_csv("scratch/najiama_out/oof_XGBOOST.csv")["OOF_Pred"].values
base_auc = roc_auc_score(y, naji_oof)
print(f"Base Najiama XGB OOF AUC: {base_auc:.7f}")

# Clip for safe logit
eps = 1e-6
p_clipped = np.clip(naji_oof, eps, 1 - eps)
l_oof = logit(p_clipped)

# Test additive shift in logit space for NO group
best_delta = 0.0
best_auc = base_auc

print("\nSweeping logit shift delta on Home_Charging_Possible == NO:")
for delta in np.linspace(-0.5, 0.5, 21):
    adj_l = l_oof.copy()
    adj_l[mask_no] += delta
    adj_p = expit(adj_l)
    auc = roc_auc_score(y, adj_p)
    print(f"  delta = {delta:+.2f}: OOF AUC = {auc:.7f} (diff = {auc - base_auc:+.7f})")
    if auc > best_auc:
        best_auc = auc
        best_delta = delta

print(f"\nBest delta: {best_delta:+.2f} with OOF AUC: {best_auc:.7f}")
