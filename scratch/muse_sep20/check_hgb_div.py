"""MUSE-SEP20 interim: HGB-B diversity vs pool (CatBoost still training)."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

train = pd.read_csv("train.csv", usecols=["id", "Will_Buy_EV"])
y = (train["Will_Buy_EV"] == "Yes").to_numpy(np.int8)
hgb = np.load("scratch/muse_sep20/oof_hgbB.npy")
mem = pd.read_csv("scratch/pull_sep18/memory_recipe_out/oof_predictions.csv")["oof_final"].to_numpy(float)
v68 = np.load("scratch/verified_ensembles/oof_v68.npy")
digit = np.load("oof_digit_features.npy")
print(f"hgb pooled={roc_auc_score(y, hgb):.7f}")
for n, a in [("memory", mem), ("v68", v68), ("digit", digit)]:
    print(f"rho(hgb,{n})={spearmanr(hgb, a)[0]:.6f}")
th = np.load("scratch/muse_sep20/test_hgbB.npy")
print(f"test finite={np.isfinite(th).all()} min={th.min():.6f} max={th.max():.6f}")
