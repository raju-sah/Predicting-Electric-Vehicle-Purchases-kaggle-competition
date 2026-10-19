import os
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from scipy.optimize import nnls

print("--- 10-MODEL META-STACKER ON FROZEN FOLDS ---")

train = pd.read_csv("train.csv")
y = (train["Will_Buy_EV"] == "Yes").astype(int).values
folds = np.load("scratch/golem_lib/folds_seed42.npy")
N_train = len(y)
N_test = 286571

# 1. Load 10 diverse models with matching OOF and TEST
models = {}

# (1) Najiama XGB
models["naji_xgb"] = {
    "oof": pd.read_csv("scratch/najiama_out/oof_XGBOOST.csv")["OOF_Pred"].values,
    "test": pd.read_csv("scratch/najiama_out/submission_XGBOOST.csv")["Will_Buy_EV"].values,
}

# (2) My GPU XGB
models["my_xgb"] = {
    "oof": pd.read_csv("scratch/my_gpu_out/oof_xgb_fixed_digits.csv")["oof_prediction"].values,
    "test": pd.read_csv("scratch/my_gpu_out/sub_xgb_fixed_digits.csv")["Will_Buy_EV"].values,
}

# (3) Deotte Simpson GPU XGB
models["deotte_xgb"] = {
    "oof": pd.read_csv("scratch/deotte_gpu_out/oof_xgb_deotte_simpson.csv")["oof_prediction"].values,
    "test": pd.read_csv("scratch/deotte_gpu_out/sub_xgb_deotte_simpson.csv")["Will_Buy_EV"].values,
}

# (4) Najiama Pure LGBM
models["naji_lgb"] = {
    "oof": pd.read_csv("scratch/outputs/najiama_pure-lgbm-model-cv-0-94606-lb-0-94637/oof_LIGHTGBM.csv")["OOF_Pred"].values,
    "test": pd.read_csv("scratch/outputs/najiama_pure-lgbm-model-cv-0-94606-lb-0-94637/submission_LIGHTGBM.csv")["Will_Buy_EV"].values,
}

# (5-10) Golem models
golem_keys = [
    ("golem_h", "oof_h.npy", "test_h.npy"),
    ("golem_i", "oof_i.npy", "test_i.npy"),
    ("golem_d_catb", "oof_d_catboost.npy", "test_d_catboost.npy"),
    ("golem_spline", "oof_h_spline_gam.npy", "test_h_spline_gam.npy"),
    ("golem_mlp", "oof_g_mlp.npy", "test_g_mlp.npy"),
    ("golem_f_spline_lr", "oof_f.npy", "test_f.npy"),
]

for name, oof_f, test_f in golem_keys:
    models[name] = {
        "oof": np.load(f"scratch/golem_lib/{oof_f}"),
        "test": np.load(f"scratch/golem_lib/{test_f}"),
    }

names = list(models.keys())
print(f"Loaded {len(names)} models:")
for name in names:
    auc = roc_auc_score(y, models[name]["oof"])
    print(f"  {name:20s}: OOF AUC = {auc:.6f}")

# Rank-transform OOF and Test features
X_oof_ranks = np.zeros((N_train, len(names)), dtype=np.float32)
X_test_ranks = np.zeros((N_test, len(names)), dtype=np.float32)

for j, name in enumerate(names):
    X_oof_ranks[:, j] = rankdata(models[name]["oof"]) / N_train
    X_test_ranks[:, j] = rankdata(models[name]["test"]) / N_test

# 2. Fit Out-Of-Fold Logistic Regression meta-model across the 5 frozen folds
oof_stack_pred = np.zeros(N_train)
test_stack_preds = np.zeros(N_test)

fold_aucs = []
print("\nFitting nested Logistic Regression meta-stacker on 5 frozen folds...")
for fold in range(5):
    val_mask = (folds == fold)
    tr_mask = ~val_mask
    
    X_tr, y_tr = X_oof_ranks[tr_mask], y[tr_mask]
    X_va, y_va = X_oof_ranks[val_mask], y[val_mask]
    
    meta = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200, random_state=42)
    meta.fit(X_tr, y_tr)
    
    va_pred = meta.predict_proba(X_va)[:, 1]
    oof_stack_pred[val_mask] = va_pred
    
    auc = roc_auc_score(y_va, va_pred)
    fold_aucs.append(auc)
    print(f"Fold {fold}: AUC = {auc:.6f} | Coeffs: {np.round(meta.coef_[0], 3)}")
    
    test_stack_preds += meta.predict_proba(X_test_ranks)[:, 1] / 5.0

total_stack_auc = roc_auc_score(y, oof_stack_pred)
print(f"\n==========================================")
print(f"10-Model Logistic Stacker OOF AUC: {total_stack_auc:.6f}")
print(f"Mean Fold AUC: {np.mean(fold_aucs):.6f} (+/- {np.std(fold_aucs):.6f})")
print(f"Single Best Model (naji_xgb):     {roc_auc_score(y, models['naji_xgb']['oof']):.6f}")
print(f"Stacker Improvement over Best:    {total_stack_auc - roc_auc_score(y, models['naji_xgb']['oof']):+.6f}")
print(f"==========================================")

# Save the stacker test predictions
np.save("scratch/test_meta_stacker_10way.npy", test_stack_preds)
np.save("scratch/oof_meta_stacker_10way.npy", oof_stack_pred)
print("Saved scratch/test_meta_stacker_10way.npy and scratch/oof_meta_stacker_10way.npy")
