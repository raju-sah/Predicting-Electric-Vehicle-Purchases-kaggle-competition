import os
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from scipy.optimize import minimize

# Load train and labels
train = pd.read_csv("train.csv")
y = (train["Will_Buy_EV"] == "Yes").astype(int).values

# Load folds
folds = np.load("scratch/golem_lib/folds_seed42.npy")

# Candidate models with matching test files in golem_lib + najiama
candidates = {}

# 1. Najiama XGB
naji_df = pd.read_csv("scratch/najiama_out/oof_XGBOOST.csv")
candidates["najiama_xgb"] = naji_df["OOF_Pred"].values

# 2. Key Golem models
golem_keys = [
    "oof_h", "oof_i", "oof_a", "oof_d_catboost", "oof_e_histgb",
    "oof_h_spline_gam", "oof_g_mlp", "oof_f", "oof_j_logreg"
]
for k in golem_keys:
    candidates[k] = np.load(f"scratch/golem_lib/{k}.npy")

print(f"Total candidates for blending: {len(candidates)}")
for name, arr in candidates.items():
    print(f"  {name:20s}: OOF AUC = {roc_auc_score(y, arr):.6f}")

# Rank-transform each candidate to [0, 1]
names = list(candidates.keys())
n_models = len(names)
R = np.zeros((len(y), n_models), dtype=np.float32)
for j, name in enumerate(names):
    R[:, j] = rankdata(candidates[name]) / len(y)

# 1. Forward stepwise hill-climbing (Caruana et al.) on CV folds
def hill_climbing_cv(R, y, folds, n_iter=50):
    n_samples, n_models = R.shape
    unique_folds = np.unique(folds)
    
    # Store overall OOF predictions from nested / fold-based blending
    oof_blend = np.zeros(n_samples)
    fold_weights = []
    
    for f in unique_folds:
        val_mask = (folds == f)
        tr_mask = ~val_mask
        
        y_tr, y_va = y[tr_mask], y[val_mask]
        R_tr, R_va = R[tr_mask], R[val_mask]
        
        # Start with best single model on training folds
        best_single = np.argmax([roc_auc_score(y_tr, R_tr[:, j]) for j in range(n_models)])
        selected = [best_single]
        cur_tr_pred = R_tr[:, best_single].copy()
        cur_va_pred = R_va[:, best_single].copy()
        
        best_tr_auc = roc_auc_score(y_tr, cur_tr_pred)
        
        for step in range(1, n_iter):
            best_cand = None
            best_cand_auc = best_tr_auc
            
            for j in range(n_models):
                cand_pred = (cur_tr_pred * len(selected) + R_tr[:, j]) / (len(selected) + 1)
                auc = roc_auc_score(y_tr, cand_pred)
                if auc > best_cand_auc:
                    best_cand_auc = auc
                    best_cand = j
            
            if best_cand is not None and (best_cand_auc - best_tr_auc) > 1e-6:
                cur_tr_pred = (cur_tr_pred * len(selected) + R_tr[:, best_cand]) / (len(selected) + 1)
                cur_va_pred = (cur_va_pred * len(selected) + R_va[:, best_cand]) / (len(selected) + 1)
                selected.append(best_cand)
                best_tr_auc = best_cand_auc
            else:
                break
        
        oof_blend[val_mask] = cur_va_pred
        
        # Count weights
        w = np.zeros(n_models)
        for idx in selected:
            w[idx] += 1.0 / len(selected)
        fold_weights.append(w)
        print(f"Fold {f}: selected {len(selected)} models, val AUC = {roc_auc_score(y_va, cur_va_pred):.6f}")
    
    total_auc = roc_auc_score(y, oof_blend)
    mean_w = np.mean(fold_weights, axis=0)
    print(f"\nNested Hill Climbing Overall OOF AUC: {total_auc:.6f}")
    print("Mean Model Weights:")
    for j, name in enumerate(names):
        if mean_w[j] > 0.001:
            print(f"  {name:20s}: {mean_w[j]:.4f}")
    
    return oof_blend, mean_w

oof_hc, weights_hc = hill_climbing_cv(R, y, folds, n_iter=60)
