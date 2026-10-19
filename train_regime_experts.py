"""Route records to purchase-regime specialists before fitting LightGBM.

`Subsidy_Available=Yes` plus `Range_Anxiety_Level=Low` contains 96% of all
positive labels but has a 29.6% positive rate. The remaining records are a
distinct low-conversion regime (~1.4%). A single global tree model must spend
capacity on both calibrations. This model uses the same raw input fields in
each branch, but independently learns their conditional response surfaces.
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


def prepare(frame: pd.DataFrame, category_levels: dict[str, pd.Index]) -> pd.DataFrame:
    out = frame[FEATURES].copy()
    for column in CATEGORIES:
        out[column] = pd.Categorical(
            out[column].fillna(category_levels[column][0]),
            categories=category_levels[column],
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

# This is a model router, not a derived feature passed into either model.
route = ((train["Subsidy_Available"] == "Yes") &
         (train["Range_Anxiety_Level"] == "Low")).to_numpy()
route_test = ((test["Subsidy_Available"] == "Yes") &
              (test["Range_Anxiety_Level"] == "Low")).to_numpy()
print(
    "primary regime rows/positives:", int(route.sum()), int(y[route].sum()),
    "other rows/positives:", int((~route).sum()), int(y[~route].sum()),
    flush=True,
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(train), dtype=np.float64)
for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
    for primary in (False, True):
        fit_idx = train_idx[route[train_idx] == primary]
        score_idx = valid_idx[route[valid_idx] == primary]
        model = lgb.LGBMClassifier(**PARAMS)
        model.fit(X.iloc[fit_idx], y[fit_idx])
        oof[score_idx] = model.predict_proba(X.iloc[score_idx])[:, 1]
    print(f"fold {fold}: OOF AUC {roc_auc_score(y[valid_idx], oof[valid_idx]):.7f}", flush=True)

auc = roc_auc_score(y, oof)
print(f"regime-expert OOF AUC: {auc:.7f}", flush=True)
np.save("oof_regime_experts.npy", oof)

pred = np.zeros(len(test), dtype=np.float64)
for primary in (False, True):
    fit_idx = np.flatnonzero(route == primary)
    score_idx = np.flatnonzero(route_test == primary)
    model = lgb.LGBMClassifier(**PARAMS)
    model.fit(X.iloc[fit_idx], y[fit_idx])
    pred[score_idx] = model.predict_proba(X_test.iloc[score_idx])[:, 1]
pd.DataFrame({"id": test["id"], TARGET: pred}).to_csv(
    "subs/regime_experts_probe.csv", index=False
)
print("wrote oof_regime_experts.npy and subs/regime_experts_probe.csv")
