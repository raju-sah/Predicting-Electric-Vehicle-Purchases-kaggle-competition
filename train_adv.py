"""Adv validation: train(1) vs test(0) from all features + id + row index, LGB 5-fold."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

FEATS = ["Age", "Annual_Income_USD", "Daily_Commute_km", "Number_of_Cars_Owned",
         "Charging_Stations_Near_Home", "Charging_Stations_Near_Work",
         "Environmental_Concern_Level", "Gender", "City_Type", "Current_Car_Type",
         "Home_Charging_Possible", "Subsidy_Available", "Range_Anxiety_Level"]
CATS = ["Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible",
        "Subsidy_Available", "Range_Anxiety_Level"]
tr = pd.read_csv("train.csv")
te = pd.read_csv("test.csv")
tr["_row"] = np.arange(len(tr))
te["_row"] = np.arange(len(te))
A = pd.concat([tr[FEATS + ["id", "_row"]], te[FEATS + ["id", "_row"]]],
              ignore_index=True)
z = np.array([1] * len(tr) + [0] * len(te))
for c in CATS:
    A[c] = A[c].fillna(A[c].mode()[0]).astype("category")
cv = StratifiedKFold(5, shuffle=True, random_state=42)
oof = np.zeros(len(A))
imp = np.zeros(len(A.columns))
for tri, val in cv.split(A, z):
    m = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=63,
                           random_state=42, verbose=-1)
    m.fit(A.iloc[tri], z[tri])
    oof[val] = m.predict_proba(A.iloc[val])[:, 1]
    imp += m.feature_importances_
print(f"ADV OOF AUC: {roc_auc_score(z, oof):.4f} (0.50 = no shift)")
for k, v in sorted(zip(A.columns, imp / 5), key=lambda t: -t[1])[:6]:
    print(" ", k, round(float(v), 1))
