"""MUSE-SEP20 model B: sklearn HistGradientBoosting on digit + novel fit-free features.

Novelty vs killed/prior work (specific new argument):
- HistGradientBoosting has never been trained in this project beyond the raw
  HGB-default baseline (v01). This is the first HGB on the digit-frame
  representation, with native categorical support (no ordinal hack).
- Same novel fit-free ratio set as CatBoost-A (disjoint from 4 killed
  interactions), letting the HGB learner exploit them with a different
  split-finding/binning algorithm (histogram, symmetric-ish growth).
- Exact gate folds StratifiedKFold(5, shuffle=True, random_state=42).

Outputs: scratch/muse_sep20/oof_hgbB.npy, test_hgbB.npy, run_hgbB.json
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

T0 = time.time()
TARGET = "Will_Buy_EV"
OUT = "scratch/muse_sep20"
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


def digit_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = {}
    for column in NUMERIC:
        values = frame[column].to_numpy(dtype=np.float64)
        for position in DIGIT_POSITIONS:
            result[f"{column}__d{position}"] = np.mod(
                np.floor(values / (10.0 ** position)), 10.0
            ).astype(np.int8)
    return pd.DataFrame(result, index=frame.index)


def build_features(train: pd.DataFrame, test: pd.DataFrame):
    feats = []
    for frame in (train, test):
        out = pd.DataFrame(index=frame.index)
        out = pd.concat([out, digit_frame(frame)], axis=1)
        income = frame["Annual_Income_USD"].to_numpy(np.float64)
        age = frame["Age"].to_numpy(np.float64)
        commute = frame["Daily_Commute_km"].to_numpy(np.float64)
        cars = frame["Number_of_Cars_Owned"].to_numpy(np.float64)
        ch_h = frame["Charging_Stations_Near_Home"].to_numpy(np.float64)
        ch_w = frame["Charging_Stations_Near_Work"].to_numpy(np.float64)
        env = frame["Environmental_Concern_Level"].to_numpy(np.float64)
        subsidy = (frame["Subsidy_Available"].astype(str) == "Yes").to_numpy(np.float64)
        home = (frame["Home_Charging_Possible"].astype(str) == "Yes").to_numpy(np.float64)
        out["f_income_per_age"] = (income / (age + 1.0)).astype(np.float32)
        out["f_chargers_total"] = (ch_h + ch_w).astype(np.float32)
        out["f_chargers_per_car"] = ((ch_h + ch_w) / (cars + 1.0)).astype(np.float32)
        out["f_commute_x_cars"] = (commute * (cars + 1.0)).astype(np.float32)
        out["f_income_x_env"] = (income / 100000.0 * env).astype(np.float32)
        out["f_env_x_subsidy"] = (env * (1.0 + subsidy)).astype(np.float32)
        out["f_work_access"] = ((ch_w + 1.0) * (1.0 + subsidy)).astype(np.float32)
        out["f_age_x_cars"] = (age * (cars + 1.0)).astype(np.float32)
        out["f_is_30k_spike"] = (np.rint(income).astype(np.int64) == 30000).astype(np.int8)
        out["f_is_millionaire_cliff"] = (income >= 170537).astype(np.int8)
        out["f_is_dead_zone"] = ((income >= 38000) & (income <= 42000)).astype(np.int8)
        out["f_is_env_hater"] = (frame["Environmental_Concern_Level"] == 1).astype(np.int8)
        out["f_home_bin"] = home.astype(np.int8)
        for c in NUMERIC:
            out[f"f_raw_{c}"] = frame[c].to_numpy(np.float64).astype(np.float32)
        for c in CATS:
            out[c] = frame[c].astype("category")
        feats.append(out)
    return feats[0], feats[1]


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
y = (train[TARGET] == "Yes").astype(np.int8).to_numpy()
X, X_test = build_features(train, test)
print(f"features: {X.shape[1]} | train {X.shape} test {X_test.shape}", flush=True)
cat_idx = [X.columns.get_loc(c) for c in CATS]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(y), np.float64)
test_pred = np.zeros(len(test), np.float64)
fold_auc = []
for fold, (fit_idx, valid_idx) in enumerate(cv.split(X, y)):
    model = HistGradientBoostingClassifier(
        max_iter=800,
        learning_rate=0.05,
        max_leaf_nodes=63,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=50,
        tol=1e-7,
        random_state=42 + fold,
        categorical_features=cat_idx,
    )
    model.fit(X.iloc[fit_idx], y[fit_idx])
    oof[valid_idx] = model.predict_proba(X.iloc[valid_idx])[:, 1]
    test_pred += model.predict_proba(X_test)[:, 1] / cv.n_splits
    auc = roc_auc_score(y[valid_idx], oof[valid_idx])
    fold_auc.append(float(auc))
    print(f"fold {fold}: AUC {auc:.7f} iters {model.n_iter_} "
          f"elapsed {(time.time()-T0)/60:.1f}min", flush=True)
    del model

pooled = float(roc_auc_score(y, oof))
np.save(f"{OUT}/oof_hgbB.npy", oof)
np.save(f"{OUT}/test_hgbB.npy", test_pred)
with open(f"{OUT}/run_hgbB.json", "w") as f:
    json.dump({"pooled_oof_auc": pooled, "fold_auc": fold_auc,
               "mean_fold_auc": float(np.mean(fold_auc)),
               "folds": "StratifiedKFold(5, shuffle=True, random_state=42)",
               "model": "HGB(max_iter=800, lr=0.05, leaves=63, minleaf=40, seeds 42+k)"},
              f, indent=2)
print(f"HGBB DONE pooled={pooled:.7f} folds={fold_auc}", flush=True)
