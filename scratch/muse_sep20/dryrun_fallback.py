"""MUSE-SEP20 dry run: validate builder math on {memory,lgb6,xgb5,hgbB} only.

No subs/ writes. Checks alignment, gate math, and rank/prob blend code paths
so the final build is a single deterministic step once catA lands.
Also serves as the fallback candidate set if catA misses the gate.
"""
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

TARGET = "Will_Buy_EV"
GATE = 0.9418 + 0.0005
train = pd.read_csv("train.csv", usecols=["id", TARGET])
sample = pd.read_csv("sample_submission.csv")
y = (train[TARGET] == "Yes").to_numpy(np.int8)
fold = np.full(len(y), -1, np.int8)
for k, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(y, y)):
    fold[va] = k


def rd(path, ids, col):
    d = pd.read_csv(path)
    assert d["id"].equals(ids), path
    return d[col].to_numpy(float)


names = ["memory", "lgb6", "xgb5", "hgbB"]
OOF = [rd("scratch/pull_sep18/memory_recipe_out/oof_predictions.csv", train["id"], "oof_final"),
       rd("scratch/pull_sep19/najiama_refresh/Pure LGBM_V6_oof.csv", train["id"], "OOF_Pred"),
       rd("scratch/pull_sep19/najiama_refresh/XGBoost_Triple_TE_5folds_oof.csv", train["id"], "OOF_Pred"),
       np.load("scratch/muse_sep20/oof_hgbB.npy")]
TEST = [rd("scratch/pull_sep18/memory_recipe_out/submission_model_only.csv", sample["id"], TARGET),
        rd("scratch/pull_sep19/najiama_refresh/Pure LGBM_V6_test.csv", sample["id"], TARGET),
        rd("scratch/pull_sep19/najiama_refresh/XGBoost_Triple_TE_5folds_test.csv", sample["id"], TARGET),
        np.load("scratch/muse_sep20/test_hgbB.npy")]


def tform(v, m):
    if m == "rank":
        return (rankdata(v, method="average") - 0.5) / len(v)
    if m == "logit":
        return logit(np.clip(v, 1e-7, 1 - 1e-7))
    return v


def blend(arrays, w, m):
    r = sum(x * a for x, a in zip(w, arrays))
    return expit(r) if m == "logit" else r


def folds(p):
    return [roc_auc_score(y[fold == k], p[fold == k]) for k in range(5)]


for label, w, m in [("hgbB-solo", [0, 0, 0, 1], "probability"),
                    ("fallback-prob", [0.45, 0.20, 0.20, 0.15], "probability"),
                    ("fallback-rank", [0.45, 0.20, 0.20, 0.15], "rank"),
                    ("fallback-logit", [0.45, 0.20, 0.20, 0.15], "logit")]:
    o = blend([tform(a, m) for a in OOF], w, m)
    t = blend([tform(a, m) for a in TEST], w, m)
    f = folds(o)
    ok = roc_auc_score(y, o) > GATE and np.isfinite(t).all()
    print(f"{label:14s} pooled={roc_auc_score(y, o):.7f} mean={np.mean(f):.7f} "
          f"qualifies={ok}")
    print("   test-rho:", {n: round(float(spearmanr(t, TEST[i])[0]), 6)
                           for i, n in enumerate(names)})
print("DRYRUN OK")
