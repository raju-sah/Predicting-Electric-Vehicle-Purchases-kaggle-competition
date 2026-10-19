"""MUSE-SEP20 staging: materialize + audit 4 provisional HGB-backed candidates.

Staged under scratch/muse_sep20/staging/ (NOT subs/) so the final four can be
promoted exactly once after the catA verdict, without overwriting submissions.
Every staged candidate gets: full format validation, fixed-fold OOF evidence,
fold-score table, test-space diversity, sha256. Weights fixed a priori.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

TARGET = "Will_Buy_EV"
GATE = 0.9418 + 0.0005
STAGE = Path("scratch/muse_sep20/staging")
STAGE.mkdir(exist_ok=True)

train = pd.read_csv("train.csv", usecols=["id", TARGET])
sample = pd.read_csv("sample_submission.csv")
test_ids = pd.read_csv("test.csv", usecols=["id"])["id"]
y = (train[TARGET] == "Yes").to_numpy(np.int8)
fold = np.full(len(y), -1, np.int8)
for k, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(y, y)):
    fold[va] = k


def rd(path, ids, col):
    d = pd.read_csv(path)
    assert d["id"].equals(ids), path
    v = d[col].to_numpy(float)
    assert np.isfinite(v).all() and ((v >= 0) & (v <= 1)).all(), path
    return v


mem_oof = pd.read_csv("scratch/pull_sep18/memory_recipe_out/oof_predictions.csv")
assert np.array_equal(mem_oof["fold"].to_numpy(), fold), "memory fold provenance"
assert np.array_equal(mem_oof["Will_Buy_EV"].to_numpy(), y), "memory labels"
names = ["memory", "lgb6", "xgb5", "hgbB"]
OOF = [mem_oof["oof_final"].to_numpy(float),
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


v57 = rd("subs/v57_public_band_swap_exact.csv", sample["id"], TARGET)
v68 = rd("subs/v68_verified5_logit.csv", sample["id"], TARGET)

variants = [
    ("p75", "muse_hgbB_solo", [0, 0, 0, 1], "probability"),
    ("p76", "muse_prob4_fixed", [0.45, 0.20, 0.20, 0.15], "probability"),
    ("p77", "muse_rank4_fixed", [0.45, 0.20, 0.20, 0.15], "rank"),
    ("p78", "muse_logit4_fixed", [0.45, 0.20, 0.20, 0.15], "logit"),
]
manifest = {"outer_split": "StratifiedKFold(5, shuffle=True, random_state=42)",
            "gate": GATE, "components": names, "candidates": [],
            "weight_note": "fixed a priori; OOF confirmatory, not selected"}
for tag, name, w, m in variants:
    o = blend([tform(a, m) for a in OOF], w, m)
    t = blend([tform(a, m) for a in TEST], w, m)
    pooled = float(roc_auc_score(y, o))
    f = [float(roc_auc_score(y[fold == k], o[fold == k])) for k in range(5)]
    fn = STAGE / f"{tag}_{name}.csv"
    frame = pd.DataFrame({"id": test_ids, TARGET: t})
    assert frame.columns.tolist() == ["id", TARGET]
    assert len(frame) == 286571 and frame["id"].is_unique
    assert frame["id"].equals(sample["id"])
    assert np.isfinite(t).all() and ((t >= 0) & (t <= 1)).all()
    frame.to_csv(fn, index=False)
    np.save(STAGE / f"oof_{tag}.npy", o)
    rec = {"stage": tag, "name": name, "method": m, "weights": w,
           "pooled_oof_auc": pooled, "mean_fold_auc": float(np.mean(f)),
           "fold_auc": f, "qualifies": bool(pooled > GATE),
           "test_rho_vs_v57": float(spearmanr(t, v57)[0]),
           "test_rho_vs_v68": float(spearmanr(t, v68)[0]),
           "test_rho_vs_memory": float(spearmanr(t, TEST[0])[0]),
           "sha256": hashlib.sha256(fn.read_bytes()).hexdigest()}
    manifest["candidates"].append(rec)
    print(json.dumps(rec), flush=True)
(STAGE / "manifest_staging.json").write_text(json.dumps(manifest, indent=2) + "\n")
print("STAGED 4/4", flush=True)
