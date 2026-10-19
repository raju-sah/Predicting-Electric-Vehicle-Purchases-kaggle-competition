"""LightGBM with fit-free decimal-position features for numeric fields.

The reported continuous values retain generator-specific quantisation detail
which default histogram binning combines.  Each decimal digit from 10^-4
through 10^3 is supplied as a separate small-integer feature.  No labels,
target encodings, interactions, or external rows enter the representation.
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
    """Extract digit k from floor(value / 10**k), without fitting anything."""
    result = {}
    for column in NUMERIC:
        values = frame[column].to_numpy(dtype=np.float64)
        for position in DIGIT_POSITIONS:
            result[f"{column}__digit_{position}"] = np.mod(
                np.floor(values / (10.0 ** position)), 10.0
            ).astype(np.int8)
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
levels = {
    column: pd.Index(pd.concat([train[column], test[column]]).dropna().unique())
    for column in CATEGORICAL
}
X = prepare(train, levels)
X_test = prepare(test, levels)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(train), dtype=np.float64)
test_pred = np.zeros(len(test), dtype=np.float64)
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
    test_pred += model.predict_proba(X_test)[:, 1] / cv.n_splits
    rounds.append(model.best_iteration_ or PARAMS["n_estimators"])
    print(
        f"fold {fold}: AUC {roc_auc_score(y[valid_idx], oof[valid_idx]):.7f} "
        f"trees {rounds[-1]}",
        flush=True,
    )

auc = roc_auc_score(y, oof)
np.save("oof_digit_features.npy", oof)
candidate = pd.DataFrame({"id": test["id"], TARGET: test_pred})
sample = pd.read_csv("sample_submission.csv", usecols=["id"])
assert candidate.columns.tolist() == ["id", TARGET]
assert candidate["id"].equals(sample["id"])
assert np.isfinite(candidate[TARGET]).all()
candidate.to_csv("subs/v08_digit_features.csv", index=False)
print(f"digit-features OOF AUC: {auc:.7f}; fold tree counts: {rounds}", flush=True)
print("wrote oof_digit_features.npy and subs/v08_digit_features.csv", flush=True)
