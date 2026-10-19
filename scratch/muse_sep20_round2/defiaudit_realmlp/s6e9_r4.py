#!/usr/bin/env python3
"""S6E9 feature round 4: pairwise TargetEncoder on categorical conjunctions.

The only untested NEW information axis (worklog Task 29): composition signal
(pair rules like anxiety=High & subsidy=No) rather than transforms of
single columns. 15 CAT_COLS pairs x 3 smoothings = +45 TE cols on top of the
standard 185 (146 base + 39 single TE).

  python s6e9_r4.py train --model xgb --fold 0
  python s6e9_r4.py compare
Artifacts in /home/z/my-project/s6e9/artifacts_v4/
"""
import argparse
import gc
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

sys.path.insert(0, "/home/z/my-project/scripts")
from s6e9_staged_v2 import get_model, add_te, CAT_COLS  # noqa: E402

ART2 = "/home/z/my-project/s6e9/artifacts_v2"
ART = "/home/z/my-project/s6e9/artifacts_v4"
os.makedirs(ART, exist_ok=True)

PAIRS = [(a, b) for i, a in enumerate(CAT_COLS) for b in CAT_COLS[i + 1:]]


def pair_frame(te_slice):
    return pd.DataFrame(
        {f"P_{a}_{b}": te_slice[a].astype(str) + "_" + te_slice[b].astype(str)
         for a, b in PAIRS},
        index=te_slice.index)


def train_one(name, fold, lr, n, es, seed=42, nfold=10):
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    X = pd.read_parquet(f"{ART2}/X_base.parquet")
    y = np.load(f"{ART2}/y.npy")
    te_src = pd.read_parquet(f"{ART2}/te_src.parquet")
    n_tr = len(X)

    skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=42)
    tri, vai = list(skf.split(X, y))[fold]

    t0 = time.time()
    # single-col TE (39 cols) exactly as baseline
    te_fit_m, te_v, te_t = add_te(te_src.iloc[tri], y[tri],
                                  [te_src.iloc[vai], te_src.iloc[n_tr:].reset_index(drop=True)])
    # pair TE (45 cols), same leakage-free protocol
    p_fit, p_v, p_t = add_te(pair_frame(te_src.iloc[tri]), y[tri],
                             [pair_frame(te_src.iloc[vai]),
                              pair_frame(te_src.iloc[n_tr:].reset_index(drop=True))])
    Xtr = np.hstack([X.iloc[tri].to_numpy(np.float32), te_fit_m, p_fit])
    Xv = np.hstack([X.iloc[vai].to_numpy(np.float32), te_v, p_v])
    X_test_full = np.hstack([X_test_base(X), te_t, p_t])
    del X, te_fit_m, te_v, te_t, p_fit, p_v, p_t
    gc.collect()

    mdl = get_model(name, lr, n, es, seed)
    if name == "lgb":
        from lightgbm import log_evaluation, early_stopping
        mdl.fit(Xtr, y[tri], eval_set=[(Xv, y[vai])],
                callbacks=[log_evaluation(0), early_stopping(es)])
    else:
        mdl.fit(Xtr, y[tri], eval_set=[(Xv, y[vai])], verbose=False)
    oof = mdl.predict_proba(Xv)[:, 1]
    pred = mdl.predict_proba(X_test_full)[:, 1]
    auc = roc_auc_score(y[vai], oof)
    bi = int(getattr(mdl, "best_iteration", 0) or getattr(mdl, "best_iteration_", 0) or 0)
    np.save(f"{ART}/oof_{name}_r4_f{fold}.npy", oof)
    np.save(f"{ART}/pred_{name}_r4_f{fold}.npy", pred)
    with open(f"{ART}/meta_{name}_r4_f{fold}.json", "w") as f:
        json.dump({"auc": float(auc), "best_iter": bi, "lr": lr, "n": n,
                   "es": es, "secs": round(time.time() - t0, 1)}, f)
    print(f"[{name} r4 fold{fold}] AUC={auc:.5f} best_iter={bi} ({time.time()-t0:.0f}s)")


def X_test_base(X):
    return pd.read_parquet(f"{ART2}/X_test_base.parquet").to_numpy(np.float32)


def compare():
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    y = np.load(f"{ART2}/y.npy")
    X = pd.read_parquet(f"{ART2}/X_base.parquet")
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    folds = list(skf.split(X, y))
    del X
    name = "xgb"
    base_auc, r4_auc, diffs = [], [], []
    for fold in range(10):
        bf = f"{ART2}/oof_{name}_n10_f{fold}.npy"
        rf = f"{ART}/oof_{name}_r4_f{fold}.npy"
        if not (os.path.exists(bf) and os.path.exists(rf)):
            continue
        _, vai = folds[fold]
        b = roc_auc_score(y[vai], np.load(bf))
        r = roc_auc_score(y[vai], np.load(rf))
        base_auc.append(b); r4_auc.append(r); diffs.append(r - b)
    if not diffs:
        print("[xgb] no r4 folds yet")
        return
    print(f"[xgb] base={np.mean(base_auc):.5f}  r4={np.mean(r4_auc):.5f}  "
          f"diff={np.mean(diffs):+.5f}  per-fold={['%+.4f' % d for d in diffs]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["train", "compare"])
    ap.add_argument("--model", default="xgb", choices=["xgb", "lgb", "cat"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--es", type=int, default=75)
    args = ap.parse_args()
    if args.stage == "compare":
        compare()
    else:
        train_one(args.model, args.fold, args.lr, args.n, args.es)
