#!/usr/bin/env python3
"""S6E9 feature round 3: threshold-distance + non-monotone interaction features.

Rationale (worklog Task 28): plateau is same-signal saturation. Monotone shifts
of recipe_score are tree-invariant, so round 3 adds ONLY non-monotone /
interaction shapes around the known cut-points:
  rec_d          = recipe_score - 5.5            (signed dist to Deotte boundary)
  rec_d_abs      = |rec_d|                        (non-monotone: distance shell)
  rec_d_x_sub    = rec_d * subsidy                (interaction surface)
  rec_d_x_home   = rec_d * home_charging
  worry_d        = commute - 5*(st_home+st_work)  (= worry_score + 150*home)
  worry_d_abs    = |worry_d|
  chargers_per_km= chargers_total / max(commute,1)
  st_imbalance   = st_home - st_work
  rec_d_abs_x_sub= rec_d_abs * subsidy
  commute_x_anxH = commute * anxiety_H

  python s6e9_r3.py features            -> artifacts_v3/X_base_r3.parquet (+test)
  python s6e9_r3.py train --model xgb --fold 0
  python s6e9_r3.py compare             -> paired OOF AUC vs baseline n10
Artifacts in /home/z/my-project/s6e9/artifacts_v3/  (y.npy + te_src reused from v2)
"""
import argparse
import json
import os
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

COMP_DIR = "/home/z/my-project/s6e9/data"
ART2 = "/home/z/my-project/s6e9/artifacts_v2"
ART = "/home/z/my-project/s6e9/artifacts_v3"
os.makedirs(ART, exist_ok=True)

CAT_COLS = ["Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible",
            "Subsidy_Available", "Range_Anxiety_Level"]
DIGIT_COLS = ["Age", "Annual_Income_USD", "Daily_Commute_km", "Number_of_Cars_Owned",
              "Charging_Stations_Near_Home", "Charging_Stations_Near_Work",
              "Environmental_Concern_Level"]
TE_RAW = DIGIT_COLS + CAT_COLS

NEW_COLS = ["rec_d", "rec_d_abs", "rec_d_x_sub", "rec_d_x_home", "worry_d",
            "worry_d_abs", "chargers_per_km", "st_imbalance", "rec_d_abs_x_sub",
            "commute_x_anxH"]


def add_r3(full):
    home = (full.Home_Charging_Possible == "Yes").astype(int)
    subsidy = (full.Subsidy_Available == "Yes").astype(int)
    anxH = (full.Range_Anxiety_Level == "High").astype(int)
    rec_d = full.recipe_score - 5.5
    worry_d = (full.Daily_Commute_km
               - 5 * full.Charging_Stations_Near_Home
               - 5 * full.Charging_Stations_Near_Work)
    full["rec_d"] = rec_d
    full["rec_d_abs"] = rec_d.abs()
    full["rec_d_x_sub"] = rec_d * subsidy
    full["rec_d_x_home"] = rec_d * home
    full["worry_d"] = worry_d
    full["worry_d_abs"] = worry_d.abs()
    full["chargers_per_km"] = (full.Charging_Stations_Near_Home
                               + full.Charging_Stations_Near_Work) / full.Daily_Commute_km.clip(lower=1)
    full["st_imbalance"] = full.Charging_Stations_Near_Home - full.Charging_Stations_Near_Work
    full["rec_d_abs_x_sub"] = rec_d.abs() * subsidy
    full["commute_x_anxH"] = full.Daily_Commute_km * anxH
    return full


def build_features():
    from scipy.stats import norm
    train = pd.read_csv(f"{COMP_DIR}/train.csv")
    test = pd.read_csv(f"{COMP_DIR}/test.csv")
    y = (train.Will_Buy_EV == "Yes").astype(int).values

    full = pd.concat([train.drop(columns=["Will_Buy_EV", "id"]),
                      test.drop(columns=["id"])], ignore_index=True)

    home = (full.Home_Charging_Possible == "Yes").astype(int)
    subsidy = (full.Subsidy_Available == "Yes").astype(int)
    full["worry_score"] = (full.Daily_Commute_km - 5 * full.Charging_Stations_Near_Home
                           - 5 * full.Charging_Stations_Near_Work - 150 * home)
    full["chargers_total"] = full.Charging_Stations_Near_Home + full.Charging_Stations_Near_Work
    full["income_x_subsidy"] = full.Annual_Income_USD / 1e5 * subsidy
    full["concern_x_subsidy"] = full.Environmental_Concern_Level * subsidy
    full["recipe_score"] = (1.2 * full.Annual_Income_USD / 1e5 + 0.6 * full.Environmental_Concern_Level
                            + 2.0 * subsidy - 1.0 * (full.Range_Anxiety_Level == "Medium")
                            - 3.0 * (full.Range_Anxiety_Level == "High"))
    p = np.clip(norm.cdf(full.recipe_score.values - 5.5), 1e-6, 1 - 1e-6)
    full["recipe_logit"] = np.log(p / (1 - p))
    full["anxiety_M"] = (full.Range_Anxiety_Level == "Medium").astype(int)
    full["anxiety_H"] = (full.Range_Anxiety_Level == "High").astype(int)

    for c in CAT_COLS:
        full[f"LE_{c}"] = full[c].astype("category").cat.codes.astype("int8")

    for col in DIGIT_COLS:
        v = full[col].fillna(0)
        for k in range(-4, 4):
            full[f"{col}_digit{k}"] = (v // (10 ** k) % 10).astype("int8")

    freq_cols = sorted(set([c for c in full.columns if "_digit" in c] +
                           DIGIT_COLS + [f"LE_{c}" for c in CAT_COLS]))
    for c in freq_cols:
        full[f"F_{c}"] = full[c].map(full[c].value_counts(normalize=True)).astype("float32")

    full = add_r3(full)

    float_cols = ["worry_score", "chargers_total", "income_x_subsidy", "concern_x_subsidy",
                  "recipe_score", "recipe_logit", "anxiety_M", "anxiety_H",
                  "Annual_Income_USD", "Daily_Commute_km", "Environmental_Concern_Level",
                  "Age", "Number_of_Cars_Owned", "Charging_Stations_Near_Home",
                  "Charging_Stations_Near_Work"] + [f"LE_{c}" for c in CAT_COLS] + NEW_COLS
    keep = float_cols + [c for c in full.columns if "_digit" in c or c.startswith("F_")]
    X = full[keep].iloc[:len(train)].reset_index(drop=True)
    X_test = full[keep].iloc[len(train):].reset_index(drop=True)
    X.to_parquet(f"{ART}/X_base_r3.parquet")
    X_test.to_parquet(f"{ART}/X_test_r3.parquet")
    if not os.path.exists(f"{ART}/y.npy"):
        np.save(f"{ART}/y.npy", y)
    print(f"r3 features saved: X={X.shape} X_test={X_test.shape} (+{len(NEW_COLS)} cols)")


def add_te(X_fit_rows, y_fit, X_apply_list):
    from sklearn.preprocessing import TargetEncoder
    out = []
    for smooth in ["auto", 10.0, 100.0]:
        enc = TargetEncoder(smooth=smooth)
        out.append(enc.fit_transform(X_fit_rows, y_fit))
        for Xa in X_apply_list:
            out.append(enc.transform(Xa))
    n_app = len(X_apply_list)
    groups = [[out[s * (n_app + 1) + i] for i in range(n_app + 1)] for s in range(3)]
    te_fit = np.hstack([g[0] for g in groups]).astype(np.float32)
    apps = [np.hstack([g[i + 1] for g in groups]).astype(np.float32)
            for i in range(n_app)]
    return [te_fit] + apps


def get_model(name, lr, n, es, seed=42):
    if name == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=n, learning_rate=lr, max_depth=7,
                             min_child_weight=10, subsample=0.9, colsample_bytree=0.9,
                             reg_alpha=0.07, reg_lambda=2.0, max_bin=256,
                             tree_method="hist", eval_metric="auc",
                             early_stopping_rounds=es, n_jobs=-1,
                             random_state=seed, verbosity=0)
    if name == "lgb":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(n_estimators=n, learning_rate=lr, num_leaves=127,
                              min_child_samples=40, subsample=0.9, colsample_bytree=0.9,
                              reg_alpha=0.07, reg_lambda=2.0,
                              eval_metric="auc", early_stopping_rounds=es,
                              n_jobs=-1, random_state=seed, verbose=-1)
    if name == "cat":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=n, learning_rate=lr, depth=7,
                                  l2_leaf_reg=3.0, eval_metric="AUC",
                                  early_stopping_rounds=es, random_seed=seed,
                                  verbose=0, allow_writing_files=False, thread_count=-1)
    raise ValueError(name)


def train_one(name, fold, lr, n, es, seed=42, nfold=10):
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    X = pd.read_parquet(f"{ART}/X_base_r3.parquet")
    X_test = pd.read_parquet(f"{ART}/X_test_r3.parquet")
    y = np.load(f"{ART}/y.npy")
    te_src = pd.read_parquet(f"{ART2}/te_src.parquet")
    n_tr = len(X)

    skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=42)
    tri, vai = list(skf.split(X, y))[fold]

    t0 = time.time()
    te_fit_m, te_v, te_t = add_te(te_src.iloc[tri], y[tri],
                                  [te_src.iloc[vai], te_src.iloc[n_tr:].reset_index(drop=True)])
    Xtr = np.hstack([X.iloc[tri].to_numpy(np.float32), te_fit_m])
    Xv = np.hstack([X.iloc[vai].to_numpy(np.float32), te_v])
    X_test_full = np.hstack([X_test.to_numpy(np.float32), te_t])

    mdl = get_model(name, lr, n, es, seed)
    tag = f"_r3" + (f"_s{seed}" if seed != 42 else "") + (f"_n{nfold}" if nfold != 5 else "")
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
    np.save(f"{ART}/oof_{name}{tag}_f{fold}.npy", oof)
    np.save(f"{ART}/pred_{name}{tag}_f{fold}.npy", pred)
    with open(f"{ART}/meta_{name}{tag}_f{fold}.json", "w") as f:
        json.dump({"auc": float(auc), "best_iter": bi, "lr": lr, "n": n,
                   "es": es, "secs": round(time.time() - t0, 1)}, f)
    print(f"[{name} r3 fold{fold}] AUC={auc:.5f} best_iter={bi} ({time.time()-t0:.0f}s)")


def compare():
    from sklearn.metrics import roc_auc_score
    y = np.load(f"{ART}/y.npy")
    from sklearn.model_selection import StratifiedKFold
    X = pd.read_parquet(f"{ART}/X_base_r3.parquet")
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    folds = list(skf.split(X, y))
    del X
    for name in ["xgb", "cat", "lgb"]:
        base_auc, r3_auc, diffs = [], [], []
        for fold in range(10):
            bf = f"{ART2}/oof_{name}_n10_f{fold}.npy"
            rf = f"{ART}/oof_{name}_r3_f{fold}.npy"
            if not (os.path.exists(bf) and os.path.exists(rf)):
                continue
            _, vai = folds[fold]
            b = roc_auc_score(y[vai], np.load(bf))
            r = roc_auc_score(y[vai], np.load(rf))
            base_auc.append(b); r3_auc.append(r); diffs.append(r - b)
        if not diffs:
            print(f"[{name}] no r3 folds yet")
            continue
        print(f"[{name}] base={np.mean(base_auc):.5f}  r3={np.mean(r3_auc):.5f}  "
              f"diff={np.mean(diffs):+.5f}  per-fold={['%+.4f' % d for d in diffs]}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["features", "train", "compare"])
    ap.add_argument("--model", default="xgb", choices=["xgb", "lgb", "cat"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--es", type=int, default=75)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--nfold", type=int, default=10)
    args = ap.parse_args()
    if args.stage == "features":
        build_features()
    elif args.stage == "compare":
        compare()
    else:
        train_one(args.model, args.fold, args.lr, args.n, args.es, args.seed, args.nfold)
