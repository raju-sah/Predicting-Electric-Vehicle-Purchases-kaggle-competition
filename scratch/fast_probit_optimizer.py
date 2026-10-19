import pandas as pd
import numpy as np
from scipy.stats import rankdata, norm
from scipy.optimize import minimize
from sklearn.metrics import roc_auc_score

print("Loading predictions...")
train = pd.read_csv('train.csv', usecols=['id', 'Will_Buy_EV'])
y_true = (train['Will_Buy_EV'] == 'Yes').astype(int).values

fv = pd.read_csv('scratch/datasets/four_views/oof_four_views.csv')['ensemble_ABCD'].values
deotte = pd.read_csv('scratch/deotte_gpu_out/oof_xgb_deotte_simpson.csv')['oof_prediction'].values
realmlp = pd.read_csv('scratch/realmlp_out/oof_preds.csv')['Will_Buy_EV'].values
miz = pd.read_csv('scratch/outputs/mizushimatoshihiko_s6e9-competition-only-lgbm-cv-0-94610-lb-0-94636/oof_predictions.csv')['Will_Buy_EV'].values

# Compute percentile ranks and probit z-scores
r_fv = (rankdata(fv) - 0.5) / len(fv)
r_deotte = (rankdata(deotte) - 0.5) / len(deotte)
r_realmlp = (rankdata(realmlp) - 0.5) / len(realmlp)
r_miz = (rankdata(miz) - 0.5) / len(miz)

z_fv = norm.ppf(np.clip(r_fv, 1e-7, 1 - 1e-7))
z_deotte = norm.ppf(np.clip(r_deotte, 1e-7, 1 - 1e-7))
z_realmlp = norm.ppf(np.clip(r_realmlp, 1e-7, 1 - 1e-7))
z_miz = norm.ppf(np.clip(r_miz, 1e-7, 1 - 1e-7))

Z = np.column_stack([z_fv, z_deotte, z_realmlp, z_miz])

# Optimize on 100k sample using Powell
np.random.seed(42)
sample_idx = np.random.choice(len(y_true), 100000, replace=False)
y_sub = y_true[sample_idx]
Z_sub = Z[sample_idx]

def loss(w):
    w = np.maximum(0, w)
    if w.sum() == 0:
        return 0
    w = w / w.sum()
    pred = Z_sub @ w
    return -roc_auc_score(y_sub, pred)

res = minimize(loss, [0.65, 0.15, 0.10, 0.10], method='Powell', options={'maxiter': 100})
opt_w = np.maximum(0, res.x)
opt_w = opt_w / opt_w.sum()

print("Optimized Probit Weights [FV, Deotte_GPU, RealMLP, Mizushima]:")
print([round(float(x), 4) for x in opt_w])

full_probit_pred = Z @ opt_w
full_auc = roc_auc_score(y_true, full_probit_pred)
print(f"Full OOF Probit AUC: {full_auc:.7f}")
print(f"Four-Views Solo AUC: {roc_auc_score(y_true, r_fv):.7f}")
print(f"Gain: {full_auc - roc_auc_score(y_true, r_fv):+.7f}")
