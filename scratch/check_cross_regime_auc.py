import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

train = pd.read_csv("train.csv")
y = (train["Will_Buy_EV"] == "Yes").astype(int).values

mask_yes = (train["Home_Charging_Possible"] == "Yes").values
mask_no = ~mask_yes

naji_oof = pd.read_csv("scratch/najiama_out/oof_XGBOOST.csv")["OOF_Pred"].values

# Cross-regime pairs:
# Pair A: YES buyer (y=1, mask_yes) vs NO non-buyer (y=0, mask_no)
y_yes_pos = naji_oof[mask_yes & (y == 1)]
y_no_neg = naji_oof[mask_no & (y == 0)]

# Pair B: NO buyer (y=1, mask_no) vs YES non-buyer (y=0, mask_yes)
y_no_pos = naji_oof[mask_no & (y == 1)]
y_yes_neg = naji_oof[mask_yes & (y == 0)]

# Compute AUC for Pair A
n_pos_A, n_neg_A = len(y_yes_pos), len(y_no_neg)
# Subsample 10000 each for fast exact calculation
np.random.seed(42)
idx_pA = np.random.choice(n_pos_A, 20000)
idx_nA = np.random.choice(n_neg_A, 20000)
auc_A = (y_yes_pos[idx_pA, None] > y_no_neg[None, idx_nA]).mean() + 0.5 * (y_yes_pos[idx_pA, None] == y_no_neg[None, idx_nA]).mean()

# Compute AUC for Pair B
n_pos_B, n_neg_B = len(y_no_pos), len(y_yes_neg)
idx_pB = np.random.choice(n_pos_B, 20000)
idx_nB = np.random.choice(n_neg_B, 20000)
auc_B = (y_no_pos[idx_pB, None] > y_yes_neg[None, idx_nB]).mean() + 0.5 * (y_no_pos[idx_pB, None] == y_yes_neg[None, idx_nB]).mean()

print(f"Cross-Regime Pair A (YES buyer vs NO non-buyer) AUC: {auc_A:.6f}")
print(f"Cross-Regime Pair B (NO buyer vs YES non-buyer) AUC:  {auc_B:.6f}")
print(f"Overall Cross-Regime AUC: {(auc_A + auc_B)/2:.6f}")
