"""Round2: disw diversity vs pool + blend potential check."""
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

y = (pd.read_csv("train.csv", usecols=["Will_Buy_EV"])["Will_Buy_EV"] == "Yes").to_numpy(np.int8)
disw = np.load("scratch/muse_sep20_round2/oof_disw.npy")
mem = pd.read_csv("scratch/pull_sep18/memory_recipe_out/oof_predictions.csv")["oof_final"].to_numpy(float)
v68 = np.load("scratch/verified_ensembles/oof_v68.npy")
digit = np.load("oof_digit_features.npy")
hgb = np.load("scratch/muse_sep20/oof_hgbB.npy")
print(f"disw pooled={roc_auc_score(y, disw):.7f}")
for n, a in [("memory", mem), ("v68", v68), ("digit", digit), ("hgbB", hgb)]:
    print(f"rho(disw,{n})={spearmanr(disw, a)[0]:.6f}")
# blend potential: 6-way rank blend quick probe (descriptive only)
all6 = [mem, v68, digit, hgb, disw,
        np.load("scratch/muse_sep20/oof_catA.npy")]
r = [(rankdata(a, method="average") - 0.5) / len(a) for a in all6]
for w in ([0.3, 0.3, 0.1, 0.1, 0.1, 0.1], [0.4, 0.2, 0.1, 0.1, 0.1, 0.1]):
    p = sum(x * a for x, a in zip(w, r))
    print(f"w={w} pooled={roc_auc_score(y, p):.7f}")
