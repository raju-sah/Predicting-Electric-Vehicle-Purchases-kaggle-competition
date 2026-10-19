"""Promote staged logit4 to subs/v79 (user-authorized submission)."""
import csv
import hashlib
import shutil

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

shutil.copy("scratch/muse_sep20/staging/p78_muse_logit4_fixed.csv",
            "subs/v79_muse_logit4_fixed.csv")
shutil.copy("scratch/muse_sep20/staging/oof_p78.npy",
            "scratch/muse_sep20_round2/oof_v79.npy")
d = pd.read_csv("subs/v79_muse_logit4_fixed.csv")
sample = pd.read_csv("sample_submission.csv")
assert d.columns.tolist() == ["id", "Will_Buy_EV"] and len(d) == 286571
assert d["id"].equals(sample["id"]) and d["id"].is_unique
v = d["Will_Buy_EV"].to_numpy(float)
assert np.isfinite(v).all() and ((v >= 0) & (v <= 1)).all()
o = np.load("scratch/muse_sep20_round2/oof_v79.npy")
y = (pd.read_csv("train.csv", usecols=["Will_Buy_EV"])["Will_Buy_EV"] == "Yes").to_numpy(np.int8)
pooled = float(roc_auc_score(y, o))
assert abs(pooled - 0.9462450440684402) < 1e-12
sha = hashlib.sha256(open("subs/v79_muse_logit4_fixed.csv", "rb").read()).hexdigest()
print(f"v79 OK pooled={pooled:.7f} sha={sha[:16]}")
with open("exp_log.csv", "a", newline="") as f:
    csv.writer(f).writerows([["v79", "Fixed logit blend mem45/lgb20/xgb20/hgb15",
                              "0.9462450", "",
                              "Promoted from staged p78 (user-authorized); strongest muse complete OOF; "
                              f"sha {sha[:16]}; SUBMITTED per user order"]])
print("logged")
