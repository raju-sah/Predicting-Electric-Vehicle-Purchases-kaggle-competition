"""MUSE-SEP20 candidate builder: v75-v78 with complete fixed-five-fold OOF.

Components (all exact StratifiedKFold(5, shuffle=True, random_state=42)):
  memory, lgb6, xgb5 (verified_ensembles provenance), catA, hgbB (new, this task).
Weights are FIXED A PRIORI (no OOF peeking): solos + one 5-way weight vector
in two pooling spaces. Reported OOF is confirmatory, not selected.
Gate: pooled OOF must exceed 0.94230, else the file is still written but the
manifest flags it NOT_QUALIFIED and the report must not recommend it.

Outputs: subs/v75..v78 + scratch/muse_sep20/oof_v7X.npy + manifest_muse.json
"""
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = "."
TARGET = "Will_Buy_EV"
OUT = "scratch/muse_sep20"
GATE = 0.9418 + 0.0005

train = pd.read_csv(f"{ROOT}/train.csv", usecols=["id", TARGET])
test = pd.read_csv(f"{ROOT}/test.csv", usecols=["id"])
sample = pd.read_csv(f"{ROOT}/sample_submission.csv")
y = (train[TARGET] == "Yes").to_numpy(np.int8)
fold = np.full(len(y), -1, np.int8)
for k, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(y, y)):
    fold[va] = k


def read_aligned(path, ids, col):
    d = pd.read_csv(path)
    assert d["id"].equals(ids), path
    v = d[col].to_numpy(float)
    assert np.isfinite(v).all() and ((v >= 0) & (v <= 1)).all(), path
    return v


comps = {}
comps["memory"] = (
    read_aligned(f"{ROOT}/scratch/pull_sep18/memory_recipe_out/oof_predictions.csv",
                 train["id"], "oof_final"),
    read_aligned(f"{ROOT}/scratch/pull_sep18/memory_recipe_out/submission_model_only.csv",
                 sample["id"], TARGET),
)
comps["lgb6"] = (
    read_aligned(f"{ROOT}/scratch/pull_sep19/najiama_refresh/Pure LGBM_V6_oof.csv",
                 train["id"], "OOF_Pred"),
    read_aligned(f"{ROOT}/scratch/pull_sep19/najiama_refresh/Pure LGBM_V6_test.csv",
                 sample["id"], TARGET),
)
comps["xgb5"] = (
    read_aligned(f"{ROOT}/scratch/pull_sep19/najiama_refresh/XGBoost_Triple_TE_5folds_oof.csv",
                 train["id"], "OOF_Pred"),
    read_aligned(f"{ROOT}/scratch/pull_sep19/najiama_refresh/XGBoost_Triple_TE_5folds_test.csv",
                 sample["id"], TARGET),
)
comps["catA"] = (np.load(f"{OUT}/oof_catA.npy"),
                 np.load(f"{OUT}/test_catA.npy"))
comps["hgbB"] = (np.load(f"{OUT}/oof_hgbB.npy"),
                 np.load(f"{OUT}/test_hgbB.npy"))
for name, (oo, tt) in comps.items():
    assert len(oo) == len(y) and len(tt) == len(sample), name
    assert np.isfinite(oo).all() and np.isfinite(tt).all(), name
assert np.array_equal(pd.read_csv(
    f"{ROOT}/scratch/pull_sep18/memory_recipe_out/oof_predictions.csv")["fold"].to_numpy(), fold)

names = ["memory", "lgb6", "xgb5", "catA", "hgbB"]
OOF = [comps[n][0] for n in names]
TEST = [comps[n][1] for n in names]


def tform(v, method):
    if method == "rank":
        return (rankdata(v, method="average") - 0.5) / len(v)
    if method == "logit":
        return logit(np.clip(v, 1e-7, 1 - 1e-7))
    return v


def blend(arrays, w, method):
    r = sum(x * a for x, a in zip(w, arrays))
    return expit(r) if method == "logit" else r


def fold_scores(p):
    return [float(roc_auc_score(y[fold == k], p[fold == k])) for k in range(5)]


W5 = [0.40, 0.15, 0.15, 0.15, 0.15]  # fixed a priori: anchor + diversity
variants = [
    ("v75", "muse_catA_solo", [0, 0, 0, 1, 0], "probability"),
    ("v76", "muse_hgbB_solo", [0, 0, 0, 0, 1], "probability"),
    ("v77", "muse_prob5_fixed", W5, "probability"),
    ("v78", "muse_rank5_fixed", W5, "rank"),
]

manifest = {"components": names, "weights5": W5,
            "outer_split": "StratifiedKFold(5, shuffle=True, random_state=42)",
            "gate": GATE, "candidates": [],
            "diversity_note": "weights fixed a priori; OOF confirmatory, not selected"}
for ver, tag, w, method in variants:
    oo = [tform(a, method) for a in OOF]
    tt = [tform(a, method) for a in TEST]
    oof_pred = blend(oo, w, method)
    test_pred = blend(tt, w, method)
    pooled = float(roc_auc_score(y, oof_pred))
    folds = fold_scores(oof_pred)
    fn = f"subs/{ver}_{tag}.csv"
    frame = sample.copy()
    frame[TARGET] = test_pred
    assert frame.columns.tolist() == ["id", TARGET]
    assert frame["id"].equals(sample["id"]) and frame["id"].is_unique
    assert np.isfinite(test_pred).all() and ((test_pred >= 0) & (test_pred <= 1)).all()
    assert len(frame) == len(sample) == 286571
    frame.to_csv(fn, index=False)
    np.save(f"{OUT}/oof_{ver}.npy", oof_pred)
    rho = {n: float(spearmanr(test_pred, comps[n][1])[0]) for n in names}
    rec = {"version": ver, "file": fn, "method": method, "weights": w,
           "pooled_oof_auc": pooled, "mean_fold_auc": float(np.mean(folds)),
           "fold_auc": folds,
           "qualifies": bool(pooled > GATE),
           "test_spearman_vs_components": rho,
           "sha256": hashlib.sha256(open(fn, "rb").read()).hexdigest()}
    manifest["candidates"].append(rec)
    print(json.dumps(rec), flush=True)

with open(f"{OUT}/manifest_muse.json", "w") as f:
    json.dump(manifest, f, indent=2)
print("manifest written", flush=True)
