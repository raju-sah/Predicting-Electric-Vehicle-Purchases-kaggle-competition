import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

train = pd.read_csv("train.csv")
y = (train["Will_Buy_EV"] == "Yes").astype(int).values

models = {
    "naji_xgb": {
        "oof": "scratch/najiama_out/oof_XGBOOST.csv",
        "oof_col": "OOF_Pred",
        "test": "scratch/najiama_out/submission_XGBOOST.csv",
    },
    "my_xgb": {
        "oof": "scratch/my_gpu_out/oof_xgb_fixed_digits.csv",
        "oof_col": "oof_prediction",
        "test": "scratch/my_gpu_out/sub_xgb_fixed_digits.csv",
    },
    "deotte_xgb": {
        "oof": "scratch/deotte_gpu_out/oof_xgb_deotte_simpson.csv",
        "oof_col": "oof_prediction",
        "test": "scratch/deotte_gpu_out/sub_xgb_deotte_simpson.csv",
    },
    "naji_lgb": {
        "oof": "scratch/outputs/najiama_pure-lgbm-model-cv-0-94606-lb-0-94637/oof_LIGHTGBM.csv",
        "oof_col": "OOF_Pred",
        "test": "scratch/outputs/najiama_pure-lgbm-model-cv-0-94606-lb-0-94637/submission_LIGHTGBM.csv",
    },
}

oofs = {}
tests = {}

for name, cfg in models.items():
    if os.path.exists(cfg["oof"]) and os.path.exists(cfg["test"]):
        df_oof = pd.read_csv(cfg["oof"])
        arr_oof = df_oof[cfg["oof_col"]].values
        auc = roc_auc_score(y, arr_oof)
        oofs[name] = arr_oof
        
        df_test = pd.read_csv(cfg["test"])
        tests[name] = df_test
        print(f"Model {name:15s}: OOF AUC = {auc:.6f}, test rows = {len(df_test)}")
    else:
        print(f"Missing files for {name}: {os.path.exists(cfg['oof'])}, {os.path.exists(cfg['test'])}")

# Check correlations between these 4 models
print("\nSpearman Rank Correlations between top 4 models:")
from scipy.stats import spearmanr
names = list(oofs.keys())
corr_matrix = np.zeros((len(names), len(names)))
for i in range(len(names)):
    for j in range(len(names)):
        corr_matrix[i, j] = spearmanr(oofs[names[i]], oofs[names[j]]).statistic
corr_df = pd.DataFrame(corr_matrix, index=names, columns=names)
print(corr_df.round(5))

# Rank each OOF to [0, 1]
ranks = {name: rankdata(arr) / len(arr) for name, arr in oofs.items()}

# Test equal rank average
avg_oof = np.mean([ranks[n] for n in names], axis=0)
auc_avg = roc_auc_score(y, avg_oof)
print(f"\nEqual Rank Average of all {len(names)} top models: OOF AUC = {auc_avg:.6f}")

# Optimize linear weights via Nelder-Mead on AUC
from scipy.optimize import minimize

def loss(weights):
    w = np.array(weights)
    w = np.maximum(w, 0)
    if w.sum() == 0:
        return 1.0
    w = w / w.sum()
    pred = sum(w[i] * ranks[names[i]] for i in range(len(names)))
    return -roc_auc_score(y, pred)

init_w = np.ones(len(names)) / len(names)
res = minimize(loss, init_w, method="Nelder-Mead", options={"maxiter": 300, "xatol": 1e-4, "fatol": 1e-7})
opt_w = np.maximum(res.x, 0)
opt_w = opt_w / opt_w.sum()

opt_pred = sum(opt_w[i] * ranks[names[i]] for i in range(len(names)))
opt_auc = roc_auc_score(y, opt_pred)
print(f"Optimized Rank Blend OOF AUC: {opt_auc:.6f} (+{opt_auc - oofs['naji_xgb'].max():.6f} vs single best)")
print("Optimal Weights:")
for i, name in enumerate(names):
    print(f"  {name:15s}: {opt_w[i]:.4f}")
