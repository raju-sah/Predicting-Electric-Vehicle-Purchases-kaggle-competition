"""Inject the published source-data purchase recipe as one continuous feature.

Competition discussion reverse-engineered the original dataset's latent score
as income + environmental concern + subsidy - range anxiety.  Trees otherwise
must approximate this oblique linear boundary with many rectangular splits.
The score is fixed from that source analysis (not fitted/tuned on competition
labels) and is evaluated with this project's prescribed seed-42 folds.
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


def recipe_score(frame: pd.DataFrame) -> pd.Series:
    """Fixed source-data formula: no competition-label-derived parameters."""
    return (
        1.2 * frame["Annual_Income_USD"] / 100_000
        + 0.6 * frame["Environmental_Concern_Level"]
        + 2.0 * (frame["Subsidy_Available"] == "Yes").astype(float)
        - 1.0 * (frame["Range_Anxiety_Level"] == "Medium").astype(float)
        - 3.0 * (frame["Range_Anxiety_Level"] == "High").astype(float)
    )


def prepare(frame: pd.DataFrame, levels: dict[str, pd.Index]) -> pd.DataFrame:
    out = frame[FEATURES].copy()
    for column in CATEGORIES:
        out[column] = pd.Categorical(
            out[column].fillna(levels[column][0]), categories=levels[column]
        )
    out["source_purchase_recipe"] = recipe_score(frame)
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
print(f"recipe alone AUC: {roc_auc_score(y, X['source_purchase_recipe']):.7f}", flush=True)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(train), dtype=np.float64)
for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
    model = lgb.LGBMClassifier(**PARAMS)
    model.fit(X.iloc[train_idx], y[train_idx])
    oof[valid_idx] = model.predict_proba(X.iloc[valid_idx])[:, 1]
    print(f"fold {fold}: OOF AUC {roc_auc_score(y[valid_idx], oof[valid_idx]):.7f}", flush=True)

auc = roc_auc_score(y, oof)
np.save("oof_recipe_feature.npy", oof)
print(f"recipe-feature OOF AUC: {auc:.7f}", flush=True)

if auc > SUBMISSION_GATE:
    model = lgb.LGBMClassifier(**PARAMS)
    model.fit(X, y)
    pred = model.predict_proba(X_test)[:, 1]
    pd.DataFrame({"id": test["id"], TARGET: pred}).to_csv(
        "subs/recipe_feature_candidate.csv", index=False
    )
    print("wrote subs/recipe_feature_candidate.csv")
else:
    print("below submission gate; full-fit and submission intentionally skipped")
