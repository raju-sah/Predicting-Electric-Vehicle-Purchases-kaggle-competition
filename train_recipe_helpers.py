"""Validate the complete generator-informed helper set with seed-42 OOF.

The original synthetic-data analysis specifies four helpers in addition to the
published purchase score: ordinal range-worry, total chargers, income times
subsidy, and concern times subsidy. These are a single source-derived
mechanism, not a search over arbitrary interactions; none overlap the four
previously killed interaction features.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

TARGET = "Will_Buy_EV"
FEATURES = [
    "Age", "Annual_Income_USD", "Daily_Commute_km", "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home", "Charging_Stations_Near_Work",
    "Environmental_Concern_Level", "Gender", "City_Type", "Current_Car_Type",
    "Home_Charging_Possible", "Subsidy_Available", "Range_Anxiety_Level",
]
CATEGORIES = [
    "Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible",
    "Subsidy_Available", "Range_Anxiety_Level",
]
PARAMS = dict(
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=63,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=5,
    random_state=42,
    verbose=-1,
)
SUBMISSION_GATE = 0.9423


def prepare(frame: pd.DataFrame, levels: dict[str, pd.Index]) -> pd.DataFrame:
    out = frame[FEATURES].copy()
    for column in CATEGORIES:
        out[column] = pd.Categorical(
            out[column].fillna(levels[column][0]), categories=levels[column]
        )
    subsidy = (frame["Subsidy_Available"] == "Yes").astype(float)
    worry = frame["Range_Anxiety_Level"].map({"Low": 0.0, "Medium": 1.0, "High": 3.0})
    out["source_range_worry"] = worry
    out["source_total_chargers"] = (
        frame["Charging_Stations_Near_Home"] + frame["Charging_Stations_Near_Work"]
    )
    out["source_income_x_subsidy"] = frame["Annual_Income_USD"] / 100_000 * subsidy
    out["source_concern_x_subsidy"] = frame["Environmental_Concern_Level"] * subsidy
    out["source_purchase_recipe"] = (
        1.2 * frame["Annual_Income_USD"] / 100_000
        + 0.6 * frame["Environmental_Concern_Level"]
        + 2.0 * subsidy
        - worry
    )
    return out


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
y = (train[TARGET] == "Yes").astype(np.int8).to_numpy()
levels = {
    column: pd.Index(pd.concat([train[column], test[column]]).dropna().unique())
    for column in CATEGORIES
}
X = prepare(train, levels)
X_test = prepare(test, levels)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(train), dtype=np.float64)
for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
    model = lgb.LGBMClassifier(**PARAMS)
    model.fit(X.iloc[train_idx], y[train_idx])
    oof[valid_idx] = model.predict_proba(X.iloc[valid_idx])[:, 1]
    print(f"fold {fold}: OOF AUC {roc_auc_score(y[valid_idx], oof[valid_idx]):.7f}", flush=True)

auc = roc_auc_score(y, oof)
np.save("oof_recipe_helpers.npy", oof)
print(f"recipe-helpers OOF AUC: {auc:.7f}", flush=True)
if auc > SUBMISSION_GATE:
    model = lgb.LGBMClassifier(**PARAMS)
    model.fit(X, y)
    pred = model.predict_proba(X_test)[:, 1]
    pd.DataFrame({"id": test["id"], TARGET: pred}).to_csv(
        "subs/recipe_helpers_candidate.csv", index=False
    )
    print("wrote subs/recipe_helpers_candidate.csv")
else:
    print("below submission gate; full-fit and submission intentionally skipped")
