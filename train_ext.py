"""Step 4 gate: LGB same 5-fold, train-only vs train+10k external."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

TARGET = "Will_Buy_EV"
FEATS = ["Age", "Annual_Income_USD", "Daily_Commute_km", "Number_of_Cars_Owned",
         "Charging_Stations_Near_Home", "Charging_Stations_Near_Work",
         "Environmental_Concern_Level", "Gender", "City_Type", "Current_Car_Type",
         "Home_Charging_Possible", "Subsidy_Available", "Range_Anxiety_Level"]
CATS = ["Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible",
        "Subsidy_Available", "Range_Anxiety_Level"]
tr = pd.read_csv("train.csv")
ex = pd.read_csv("external/EV_Adoption_and_Range_Anxiety_Dataset.csv").drop(columns=["Buyer_ID"])
ex = ex[ex[TARGET].isin(["Yes", "No"])]
for c in ["Annual_Income_USD", "Daily_Commute_km", "Environmental_Concern_Level"]:
    ex[c] = ex[c].fillna(tr[c].median())  # ponytail: train median, ~2% missing
y = (tr[TARGET] == "Yes").astype(int).to_numpy()
X = tr[FEATS].copy()
Xe = ex[FEATS].copy()
ye = (ex[TARGET] == "Yes").astype(int).to_numpy()
for c in CATS:
    m = X[c].mode()[0]
    X[c] = X[c].fillna(m).astype("category")
    Xe[c] = Xe[c].fillna(m).astype("category")

P = dict(n_estimators=1000, learning_rate=0.03, num_leaves=63, feature_fraction=0.8,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=5, random_state=42, verbose=-1)
cv = StratifiedKFold(5, shuffle=True, random_state=42)
for name, merge in [("base", False), ("+ext", True)]:
    oof = np.zeros(len(y))
    for tri, val in cv.split(X, y):
        Xf = pd.concat([X.iloc[tri], Xe]) if merge else X.iloc[tri]
        yf = np.concatenate([y[tri], ye]) if merge else y[tri]
        m = lgb.LGBMClassifier(**P)
        m.fit(Xf, yf)
        oof[val] = m.predict_proba(X.iloc[val])[:, 1]
    print(f"{name} OOF: {roc_auc_score(y, oof):.4f}", flush=True)
