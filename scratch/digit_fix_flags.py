"""Digit-precision fix + thread clip flags, same folds/params as train_digit_features.py.
Fix: integer pre-scaling (rint x1e4) instead of float floor-divide — old code
floored ~90% of sub-unit digits one low. Flags are fit-free, from public
thread 739303 discussion (30000 clip, 170537 ceiling, 38-42k dead zone,
commute 5.0 cluster). OOF only, no submission.
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
DIGIT_POSITIONS = range(-4, 4)
PARAMS = dict(
    n_estimators=3000,
    learning_rate=0.03,
    num_leaves=63,
    min_child_samples=40,
    subsample=0.9,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    max_bin=255,
    random_state=42,
    verbose=-1,
)


def digit_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = {}
    for column in NUMERIC:
        scaled = np.rint(frame[column].to_numpy(dtype=np.float64) * 1e4).astype(np.int64)
        for position in DIGIT_POSITIONS:
            result[f"{column}__digit_{position}"] = (
                (scaled // (10 ** (position + 4))) % 10
            ).astype(np.int8)
    return pd.DataFrame(result, index=frame.index)


def clip_flags(frame: pd.DataFrame) -> pd.DataFrame:
    income = frame["Annual_Income_USD"].to_numpy(dtype=np.float64)
    commute = frame["Daily_Commute_km"].to_numpy(dtype=np.float64)
    return pd.DataFrame(
        {
            "flag_income_eq_30000": (income == 30000.0).astype(np.int8),
            "flag_income_ge_170537": (income >= 170537.0).astype(np.int8),
            "flag_income_deadzone_38_42": ((income >= 38000.0) & (income <= 42000.0)).astype(np.int8),
            "flag_commute_eq_5": (commute == 5.0).astype(np.int8),
        },
        index=frame.index,
    )


def prepare(frame: pd.DataFrame, category_levels: dict[str, pd.Index]) -> pd.DataFrame:
    result = frame[NUMERIC + CATEGORICAL].copy()
    for column in CATEGORICAL:
        result[column] = pd.Categorical(
            result[column].fillna(category_levels[column][0]),
            categories=category_levels[column],
        )
    return pd.concat([result, digit_frame(frame), clip_flags(frame)], axis=1)


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
y = (train[TARGET] == "Yes").astype(np.int8).to_numpy()
levels = {
    column: pd.Index(pd.concat([train[column], test[column]]).dropna().unique())
    for column in CATEGORICAL
}
X = prepare(train, levels)
X_test = prepare(test, levels)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(train), dtype=np.float64)
rounds = []

for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
    model = lgb.LGBMClassifier(**PARAMS)
    model.fit(
        X.iloc[train_idx],
        y[train_idx],
        eval_set=[(X.iloc[valid_idx], y[valid_idx])],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(100, verbose=False)],
    )
    oof[valid_idx] = model.predict_proba(X.iloc[valid_idx])[:, 1]
    rounds.append(model.best_iteration_ or PARAMS["n_estimators"])
    print(
        f"fold {fold}: AUC {roc_auc_score(y[valid_idx], oof[valid_idx]):.7f} "
        f"trees {rounds[-1]}",
        flush=True,
    )

auc = roc_auc_score(y, oof)
np.save("scratch/oof_digit_fix.npy", oof)
print(f"digit-fix+flags OOF AUC: {auc:.7f} (baseline digit 0.9438); trees: {rounds}", flush=True)
