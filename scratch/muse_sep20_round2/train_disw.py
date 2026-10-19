"""Round2 model C: disagreement-weighted LightGBM (NEW mechanism).

Idea: GBM blends are saturated because every learner minimizes the same global
loss. Reweight training rows by cross-model disagreement so the learner spends
capacity where the pool is uncertain. Never tried in this project (killed list
covers interactions/TE/external/pseudo/GMM/RF — not sample reweighting).

Leak safety: weights derive ONLY from out-of-fold predictions of frozen
components (memory/lgb6/xgb5 ranks), never from labels. OOF vectors are
already out-of-fold, so full-vector weights are label-leak-free.
Weight strength k=3 fixed a priori; single run, no tuning, no peeking.

Features: proven digit frame + raw numerics + native cats (digit solo 0.9438).
Folds: exact StratifiedKFold(5, shuffle=True, random_state=42).
Outputs: scratch/muse_sep20_round2/oof_disw.npy, test_disw.npy, run_disw.json
"""
import json
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

T0 = time.time()
TARGET = "Will_Buy_EV"
OUT = "scratch/muse_sep20_round2"
NUMERIC = [
    "Age", "Annual_Income_USD", "Daily_Commute_km", "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home", "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
]
CATS = [
    "Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible",
    "Subsidy_Available", "Range_Anxiety_Level",
]
DIGIT_POSITIONS = range(-4, 4)
K = 3.0  # fixed a priori disagreement strength


def digit_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = {}
    for column in NUMERIC:
        values = frame[column].to_numpy(dtype=np.float64)
        for position in DIGIT_POSITIONS:
            result[f"{column}__d{position}"] = np.mod(
                np.floor(values / (10.0 ** position)), 10.0).astype(np.int8)
    return pd.DataFrame(result, index=frame.index)


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
y = (train[TARGET] == "Yes").astype(np.int8).to_numpy()

mem = pd.read_csv("scratch/pull_sep18/memory_recipe_out/oof_predictions.csv")
lgb6 = pd.read_csv("scratch/pull_sep19/najiama_refresh/Pure LGBM_V6_oof.csv")
xgb5 = pd.read_csv("scratch/pull_sep19/najiama_refresh/XGBoost_Triple_TE_5folds_oof.csv")
assert (mem["id"] == train["id"]).all()
ranks = np.column_stack([
    rankdata(mem["oof_final"].to_numpy(float)) / len(mem),
    rankdata(lgb6["OOF_Pred"].to_numpy(float)) / len(lgb6),
    rankdata(xgb5["OOF_Pred"].to_numpy(float)) / len(xgb5),
])
disagreement = ranks.std(axis=1)
# Heavy-tailed: raw multiples peak at 63x. Clip to [1,5] — preserves the
# mechanism (upweight uncertain rows) without outlier pathology. Still fully
# a priori: no OOF/label peeking involved in this fix.
weights = np.clip(1.0 + K * (disagreement / disagreement.mean()), 1.0, 5.0)
print(f"weight range {weights.min():.3f}-{weights.max():.3f} mean {weights.mean():.3f}",
      flush=True)

levels = {c: pd.Index(pd.concat([train[c], test[c]]).dropna().unique()) for c in CATS}


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[NUMERIC + CATS].copy()
    for column in CATS:
        result[column] = pd.Categorical(result[column].fillna(levels[column][0]),
                                        categories=levels[column])
    return pd.concat([result, digit_frame(frame)], axis=1)


X = prepare(train)
X_test = prepare(test)
print(f"features {X.shape[1]}", flush=True)

PARAMS = dict(n_estimators=3000, learning_rate=0.03, num_leaves=63,
              min_child_samples=40, subsample=0.9, subsample_freq=1,
              colsample_bytree=0.8, reg_lambda=1.0, max_bin=255,
              random_state=42, verbose=-1)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(y), np.float64)
test_pred = np.zeros(len(test), np.float64)
fold_auc = []
for fold, (fit_idx, valid_idx) in enumerate(cv.split(X, y)):
    model = lgb.LGBMClassifier(**PARAMS)
    model.fit(X.iloc[fit_idx], y[fit_idx], sample_weight=weights[fit_idx],
              eval_set=[(X.iloc[valid_idx], y[valid_idx])],
              eval_metric="auc",
              callbacks=[lgb.early_stopping(100, verbose=False)])
    oof[valid_idx] = model.predict_proba(X.iloc[valid_idx])[:, 1]
    test_pred += model.predict_proba(X_test)[:, 1] / cv.n_splits
    auc = roc_auc_score(y[valid_idx], oof[valid_idx])
    fold_auc.append(float(auc))
    print(f"fold {fold}: AUC {auc:.7f} trees {model.best_iteration_} "
          f"elapsed {(time.time()-T0)/60:.1f}min", flush=True)

pooled = float(roc_auc_score(y, oof))
np.save(f"{OUT}/oof_disw.npy", oof)
np.save(f"{OUT}/test_disw.npy", test_pred)
json.dump({"pooled_oof_auc": pooled, "fold_auc": fold_auc,
           "mean_fold_auc": float(np.mean(fold_auc)),
           "folds": "StratifiedKFold(5, shuffle=True, random_state=42)",
           "mechanism": "disagreement-weighted LGBM k=3 (a priori)"},
          open(f"{OUT}/run_disw.json", "w"), indent=2)
print(f"DISW DONE pooled={pooled:.7f}", flush=True)
