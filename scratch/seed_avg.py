"""Seed-average within family: fixed digit frame, model seeds 42/43/44, same seed-42 folds.
Raw-mean OOF (forum heuristic: raw WITHIN family). Compares vs solo 0.9438048.
OOF only, no submission.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

TARGET = "Will_Buy_EV"
NUMERIC = [
    "Age", "Annual_Income_USD", "Daily_Commute_km", "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home", "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
]
CATEGORICAL = [
    "Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible",
    "Subsidy_Available", "Range_Anxiety_Level",
]
SEEDS = (42, 43, 44)


def base_params(seed):
    return dict(
        n_estimators=3000, learning_rate=0.03, num_leaves=63,
        min_child_samples=40, subsample=0.9, subsample_freq=1,
        colsample_bytree=0.8, reg_lambda=1.0, max_bin=255,
        random_state=seed, verbose=-1,
    )


def digit_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = {}
    for column in NUMERIC:
        scaled = np.rint(frame[column].to_numpy(dtype=np.float64) * 1e4).astype(np.int64)
        for position in range(-4, 4):
            result[f"{column}__digit_{position}"] = ((scaled // (10 ** (position + 4))) % 10).astype(np.int8)
    return pd.DataFrame(result, index=frame.index)


def prepare(frame: pd.DataFrame, category_levels: dict[str, pd.Index]) -> pd.DataFrame:
    result = frame[NUMERIC + CATEGORICAL].copy()
    for column in CATEGORICAL:
        result[column] = pd.Categorical(
            result[column].fillna(category_levels[column][0]),
            categories=category_levels[column],
        )
    return pd.concat([result, digit_frame(frame)], axis=1)


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
y = (train[TARGET] == "Yes").astype(np.int8).to_numpy()
levels = {c: pd.Index(pd.concat([train[c], test[c]]).dropna().unique()) for c in CATEGORICAL}
X = prepare(train, levels)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_seeds = np.zeros((len(SEEDS), len(train)))

for s, seed in enumerate(SEEDS):
    for fold, (tri, vai) in enumerate(cv.split(X, y), start=1):
        model = lgb.LGBMClassifier(**base_params(seed))
        model.fit(X.iloc[tri], y[tri], eval_set=[(X.iloc[vai], y[vai])],
                  eval_metric="auc", callbacks=[lgb.early_stopping(100, verbose=False)])
        oof_seeds[s, vai] = model.predict_proba(X.iloc[vai])[:, 1]
    print(f"seed {seed}: OOF {roc_auc_score(y, oof_seeds[s]):.7f}", flush=True)

mean_oof = oof_seeds.mean(axis=0)
print(f"seed-mean OOF: {roc_auc_score(y, mean_oof):.7f} (solo-fix 0.9438048)", flush=True)
np.save("scratch/oof_seedavg.npy", mean_oof)
