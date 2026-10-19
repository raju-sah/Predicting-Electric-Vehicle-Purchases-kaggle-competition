"""Round2 model E: focal-loss LightGBM on digit frame (NEW local mechanism).

Najiama's public V3+focal blend reports OOF 0.94613 / public 0.94638, but no
focal objective has ever been trained in this project (all local GBMs use
plain binary logloss). Focal loss down-weights easy rows, forcing capacity
onto boundary rows — a different optimization target, hence potentially
different rankings. gamma=2.0, alpha=0.25 fixed a priori; single run.
Features: digit frame + raw + native cats. Exact gate folds.
Outputs: scratch/muse_sep20_round2/oof_focal.npy, test_focal.npy, run_focal.json
"""
import json
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
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
CATS = ["Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible",
        "Subsidy_Available", "Range_Anxiety_Level"]
DIGIT_POSITIONS = range(-4, 4)
GAMMA, ALPHA = 2.0, 0.25  # fixed a priori


def focal_obj(y_pred, data):
    y = data.get_label()
    p = 1.0 / (1.0 + np.exp(-y_pred))
    p = np.clip(p, 1e-7, 1 - 1e-7)
    pt = np.where(y == 1, p, 1 - p)
    w = ALPHA * (1 - pt) ** GAMMA
    grad = w * (p - y)
    hess = w * p * (1 - p) * (GAMMA * np.abs(p - y) / np.maximum(pt, 1e-7) + 1.0)
    return grad, hess


def focal_feval(y_pred, data):
    from sklearn.metrics import roc_auc_score as _auc
    y = data.get_label()
    p = 1.0 / (1.0 + np.exp(-y_pred))
    return "auc", _auc(y, p), True


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
levels = {c: pd.Index(pd.concat([train[c], test[c]]).dropna().unique()) for c in CATS}


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame[NUMERIC + CATS].copy()
    for column in CATS:
        result[column] = pd.Categorical(result[column].fillna(levels[column][0]),
                                        categories=levels[column])
    return pd.concat([result, digit_frame(frame)], axis=1)


X = prepare(train)
X_test = prepare(test)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(y), np.float64)
test_pred = np.zeros(len(test), np.float64)
fold_auc = []
for fold, (fit_idx, valid_idx) in enumerate(cv.split(X, y)):
    d_fit = lgb.Dataset(X.iloc[fit_idx], label=y[fit_idx])
    d_valid = lgb.Dataset(X.iloc[valid_idx], label=y[valid_idx], reference=d_fit)
    model = lgb.train(dict(objective=focal_obj, learning_rate=0.03, num_leaves=63,
                           min_child_samples=40, feature_fraction=0.8,
                           bagging_fraction=0.9, bagging_freq=1, lambda_l2=1.0,
                           max_bin=255, seed=42 + fold, verbose=-1),
                      d_fit, num_boost_round=3000, valid_sets=[d_valid],
                      feval=focal_feval,
                      callbacks=[lgb.early_stopping(100, verbose=False)])
    oof[valid_idx] = model.predict(X.iloc[valid_idx])
    test_pred += model.predict(X_test) / cv.n_splits
    auc = roc_auc_score(y[valid_idx], oof[valid_idx])
    fold_auc.append(float(auc))
    print(f"fold {fold}: AUC {auc:.7f} trees {model.best_iteration} "
          f"elapsed {(time.time()-T0)/60:.1f}min", flush=True)

pooled = float(roc_auc_score(y, oof))
np.save(f"{OUT}/oof_focal.npy", oof)
np.save(f"{OUT}/test_focal.npy", test_pred)
json.dump({"pooled_oof_auc": pooled, "fold_auc": fold_auc,
           "mean_fold_auc": float(np.mean(fold_auc)),
           "folds": "StratifiedKFold(5, shuffle=True, random_state=42)",
           "mechanism": "LGBM custom focal objective gamma=2 alpha=0.25 (a priori)"},
          open(f"{OUT}/run_focal.json", "w"), indent=2)
print(f"FOCAL DONE pooled={pooled:.7f}", flush=True)
