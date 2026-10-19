"""Step 6 gate: pseudo-label honesty check. LGB same 5-fold, base vs +confident pseudo."""
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
te = pd.read_csv("test.csv")
v04 = pd.read_csv("subs/v04_4way.csv")
assert (v04["id"] == te["id"]).all()
y = (tr[TARGET] == "Yes").astype(int).to_numpy()
X = tr[FEATS].copy()
Xe = te[FEATS].copy()
for c in CATS:
    m = X[c].mode()[0]
    X[c] = X[c].fillna(m).astype("category")
    Xe[c] = Xe[c].fillna(m).astype("category")
p = v04[TARGET].to_numpy()
conf = (p < 0.03) | (p > 0.97)
print(f"confident: {conf.sum()} / {len(p)}")
Xpl, ypl = Xe[conf], (p[conf] > 0.5).astype(int)
P = dict(n_estimators=1000, learning_rate=0.03, num_leaves=63, feature_fraction=0.8,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=5, random_state=42, verbose=-1)
cv = StratifiedKFold(5, shuffle=True, random_state=42)
for name, use_pl in [("base", False), ("+pseudo", True)]:
    oof = np.zeros(len(y))
    for tri, val in cv.split(X, y):
        Xf = pd.concat([X.iloc[tri], Xpl]) if use_pl else X.iloc[tri]
        yf = np.concatenate([y[tri], ypl]) if use_pl else y[tri]
        m = lgb.LGBMClassifier(**P)
        m.fit(Xf, yf)
        oof[val] = m.predict_proba(X.iloc[val])[:, 1]
    print(f"{name} OOF: {roc_auc_score(y, oof):.4f}", flush=True)
