import pandas as pd
import numpy as np
from scipy.stats import rankdata, norm
from sklearn.metrics import roc_auc_score

print("Loading OOF predictions...")
train = pd.read_csv('train.csv', usecols=['id', 'Will_Buy_EV'])
y_true = (train['Will_Buy_EV'] == 'Yes').astype(int).values

fv = pd.read_csv('scratch/datasets/four_views/oof_four_views.csv')['ensemble_ABCD'].values
deotte = pd.read_csv('scratch/deotte_gpu_out/oof_xgb_deotte_simpson.csv')['oof_prediction'].values
digits = pd.read_csv('scratch/my_gpu_out/oof_xgb_fixed_digits.csv')['oof_prediction'].values
realmlp = pd.read_csv('scratch/realmlp_out/oof_preds.csv')['Will_Buy_EV'].values
miz = pd.read_csv('scratch/outputs/mizushimatoshihiko_s6e9-competition-only-lgbm-cv-0-94610-lb-0-94636/oof_predictions.csv')['Will_Buy_EV'].values

# Compute percentile ranks
r_fv = (rankdata(fv) - 0.5) / len(fv)
r_deotte = (rankdata(deotte) - 0.5) / len(deotte)
r_digits = (rankdata(digits) - 0.5) / len(digits)
r_realmlp = (rankdata(realmlp) - 0.5) / len(realmlp)
r_miz = (rankdata(miz) - 0.5) / len(miz)

print(f"Individual OOF AUCs:")
print(f"Four-Views:     {roc_auc_score(y_true, r_fv):.7f}")
print(f"Deotte GPU XGB: {roc_auc_score(y_true, r_deotte):.7f}")
print(f"Digits GPU XGB: {roc_auc_score(y_true, r_digits):.7f}")
print(f"RealMLP NN:     {roc_auc_score(y_true, r_realmlp):.7f}")
print(f"Mizushima LGBM: {roc_auc_score(y_true, r_miz):.7f}")

# 1. Standard Linear Rank Blend
w = [0.65, 0.15, 0.00, 0.10, 0.10]
linear_blend = (w[0]*r_fv + w[1]*r_deotte + w[2]*r_digits + w[3]*r_realmlp + w[4]*r_miz)
print(f"\nLinear Rank Blend AUC: {roc_auc_score(y_true, linear_blend):.7f}")

# 2. Probit / Rank Gauss Blend
# Convert percentile ranks to standard normal quantiles
z_fv = norm.ppf(np.clip(r_fv, 1e-7, 1 - 1e-7))
z_deotte = norm.ppf(np.clip(r_deotte, 1e-7, 1 - 1e-7))
z_digits = norm.ppf(np.clip(r_digits, 1e-7, 1 - 1e-7))
z_realmlp = norm.ppf(np.clip(r_realmlp, 1e-7, 1 - 1e-7))
z_miz = norm.ppf(np.clip(r_miz, 1e-7, 1 - 1e-7))

probit_blend = (w[0]*z_fv + w[1]*z_deotte + w[2]*z_digits + w[3]*z_realmlp + w[4]*z_miz)
print(f"Probit Rank Gauss Blend AUC: {roc_auc_score(y_true, probit_blend):.7f}")
print(f"Delta Probit vs Linear: {roc_auc_score(y_true, probit_blend) - roc_auc_score(y_true, linear_blend):+.7f}")

# 3. Grid search optimal weights for Probit Blend
best_auc = 0
best_weights = None
for w_fv in np.linspace(0.40, 0.70, 7):
    for w_deotte in np.linspace(0.10, 0.30, 5):
        for w_real in np.linspace(0.05, 0.20, 4):
            for w_miz in np.linspace(0.05, 0.20, 4):
                tot = w_fv + w_deotte + w_real + w_miz
                z = (w_fv*z_fv + w_deotte*z_deotte + w_real*z_realmlp + w_miz*z_miz) / tot
                score = roc_auc_score(y_true, z)
                if score > best_auc:
                    best_auc = score
                    best_weights = (w_fv/tot, w_deotte/tot, w_real/tot, w_miz/tot)

print(f"\nBest Probit Blend OOF AUC: {best_auc:.7f}")
print(f"Best Weights (FV, Deotte, RealMLP, Mizushima): {[round(x, 4) for x in best_weights]}")
