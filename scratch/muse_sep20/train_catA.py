"""MUSE-SEP20 model A: CatBoost ordered boosting on digit + novel fit-free features.

Novelty vs killed/prior work (specific new argument):
- Early cbm_native (0.9414) used raw features only. This is the first CatBoost
  trained on the proven digit-frame representation + memory-threshold flags.
- Ordered boosting is a fundamentally different inductive bias from the
  LGBM/XGB-dominated pool (all current verified components are LGBM/XGB).
- Interactions are fit-free ratios, disjoint from the 4 killed interactions
  (charging_access, anx_commute, income_per_car, home_sub) which are excluded.
- Single seed, depth 6, early stopping; exact gate folds
  StratifiedKFold(5, shuffle=True, random_state=42).

Outputs (all under scratch/muse_sep20/):
  oof_catA.npy, test_catA.npy, run_catA.json
"""
import json
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

T0 = time.time()
TARGET = "Will_Buy_EV"
ID = "id"
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
        # Novel fit-free ratios (disjoint from the 4 killed interactions).
        out["f_income_per_age"] = (income / (age + 1.0)).astype(np.float32)
        out["f_chargers_total"] = (ch_h + ch_w).astype(np.float32)
        out["f_chargers_per_car"] = ((ch_h + ch_w) / (cars + 1.0)).astype(np.float32)
        out["f_commute_x_cars"] = (commute * (cars + 1.0)).astype(np.float32)
        out["f_income_x_env"] = (income / 100000.0 * env).astype(np.float32)
        out["f_env_x_subsidy"] = (env * (1.0 + subsidy)).astype(np.float32)
        out["f_work_access"] = ((ch_w + 1.0) * (1.0 + subsidy)).astype(np.float32)
        out["f_age_x_cars"] = (age * (cars + 1.0)).astype(np.float32)
        # Memory-threshold flags from prior EDA (uniform application, no fold leak).
        out["f_is_30k_spike"] = (np.rint(income).astype(np.int64) == 30000).astype(np.int8)
        out["f_is_millionaire_cliff"] = (income >= 170537).astype(np.int8)
        out["f_is_dead_zone"] = ((income >= 38000) & (income <= 42000)).astype(np.int8)
        out["f_is_env_hater"] = (frame["Environmental_Concern_Level"] == 1).astype(np.int8)
        out["f_home_bin"] = home.astype(np.int8)
        for c in NUMERIC:
            out[f"f_raw_{c}"] = frame[c].to_numpy(np.float64).astype(np.float32)
        for c in CATS:
            out[c] = frame[c].astype(str)
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
    model = CatBoostClassifier(
        iterations=2500,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=5.0,
        random_seed=42 + fold,
        early_stopping_rounds=150,
        task_type="CPU",
        thread_count=6,
        verbose=250,
    )
    model.fit(
        Pool(X.iloc[fit_idx], y[fit_idx], cat_features=cat_idx),
        eval_set=Pool(X.iloc[valid_idx], y[valid_idx], cat_features=cat_idx),
        use_best_model=True,
    )
    oof[valid_idx] = model.predict_proba(X.iloc[valid_idx])[:, 1]
    test_pred += model.predict_proba(X_test)[:, 1] / cv.n_splits
    auc = roc_auc_score(y[valid_idx], oof[valid_idx])
    fold_auc.append(float(auc))
    print(f"fold {fold}: AUC {auc:.7f} trees {model.tree_count_} "
          f"elapsed {(time.time()-T0)/60:.1f}min", flush=True)
    del model

pooled = float(roc_auc_score(y, oof))
np.save(f"{OUT}/oof_catA.npy", oof)
np.save(f"{OUT}/test_catA.npy", test_pred)
with open(f"{OUT}/run_catA.json", "w") as f:
    json.dump({"pooled_oof_auc": pooled, "fold_auc": fold_auc,
               "mean_fold_auc": float(np.mean(fold_auc)),
               "folds": "StratifiedKFold(5, shuffle=True, random_state=42)",
               "model": "CatBoostClassifier(iter=2500, lr=0.03, depth=6, l2=5, seeds 42+k)"},
              f, indent=2)
print(f"CATA DONE pooled={pooled:.7f} folds={fold_auc}", flush=True)
