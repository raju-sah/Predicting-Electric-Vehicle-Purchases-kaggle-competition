"""MUSE-SEP20 final validator: re-verify subs/v75-v78 + OOFs from disk."""
import hashlib
import json

import numpy as np
import pandas as pd
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
man = json.load(open("scratch/muse_sep20/manifest_muse.json"))
ok = True
for rec in man["candidates"]:
    fn = rec["file"]
    d = pd.read_csv(fn)
    assert d.columns.tolist() == ["id", TARGET], fn
    assert len(d) == 286571 and d["id"].is_unique and d["id"].equals(sample["id"]), fn
    v = d[TARGET].to_numpy(float)
    assert np.isfinite(v).all() and ((v >= 0) & (v <= 1)).all(), fn
    sha = hashlib.sha256(open(fn, "rb").read()).hexdigest()
    assert sha == rec["sha256"], fn
    o = np.load(f"scratch/muse_sep20/oof_{rec['version']}.npy")
    pooled = float(roc_auc_score(y, o))
    assert abs(pooled - rec["pooled_oof_auc"]) < 1e-12, fn
    assert pooled > GATE, fn
    per = [round(float(roc_auc_score(y[fold == k], o[fold == k])), 7) for k in range(5)]
    print(f"{rec['version']} {fn}: rows/id/finite/bounds/sha/OOF-gate OK "
          f"pooled={pooled:.7f} folds={per}")
print("ALL FOUR VALIDATED")
