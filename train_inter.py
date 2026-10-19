"""Step 3: interaction FE gated on LGB OOF (base 0.9418). argv: set of feats to add."""
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

TARGET = "Will_Buy_EV"
ORD = {"Low": 0, "Medium": 1, "High": 2}
CATS = ["Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible",
        "Subsidy_Available"]
ALL = ["charging_access", "anx_commute", "income_per_car", "home_sub"]
want = sys.argv[1:] or ALL
train = pd.read_csv("train.csv")
y = (train[TARGET] == "Yes").astype(int).to_numpy()
X = train[["Age", "Annual_Income_USD", "Daily_Commute_km", "Number_of_Cars_Owned",
           "Charging_Stations_Near_Home", "Charging_Stations_Near_Work",
           "Environmental_Concern_Level"] + CATS].copy()
X["Range_Anx_ord"] = train["Range_Anxiety_Level"].map(ORD)
if "charging_access" in want:
    X["charging_access"] = train["Charging_Stations_Near_Home"] + train["Charging_Stations_Near_Work"]
if "anx_commute" in want:
    X["anx_commute"] = X["Range_Anx_ord"] * train["Daily_Commute_km"]
if "income_per_car" in want:
    X["income_per_car"] = train["Annual_Income_USD"] / (train["Number_of_Cars_Owned"] + 1)
if "home_sub" in want:
    X["home_sub"] = ((train["Home_Charging_Possible"] == "Yes") &
                     (train["Subsidy_Available"] == "Yes")).astype(int)
for c in CATS:
    X[c] = X[c].fillna(X[c].mode()[0]).astype("category")
cv = StratifiedKFold(5, shuffle=True, random_state=42)
oof = np.zeros(len(y))
for tri, val in cv.split(X, y):
    m = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.03, num_leaves=63,
                           feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                           lambda_l2=5, random_state=42, verbose=-1)
    m.fit(X.iloc[tri], y[tri])
    oof[val] = m.predict_proba(X.iloc[val])[:, 1]
print(f"feats={want} LGB OOF: {roc_auc_score(y, oof):.4f}")
np.save(f"oof_inter_{'+'.join(want) if len(want) < 4 else 'all'}.npy", oof)
