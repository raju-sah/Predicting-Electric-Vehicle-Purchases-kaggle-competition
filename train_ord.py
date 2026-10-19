"""Step 2: ordinal Range_Anxiety (Low0/Med1/High2), LGB 5-fold, rank check."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

TARGET = "Will_Buy_EV"
ORD = {"Low": 0, "Medium": 1, "High": 2}
CATS = ["Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible",
        "Subsidy_Available"]
train = pd.read_csv("train.csv")
y = (train[TARGET] == "Yes").astype(int).to_numpy()
X = train[["Age", "Annual_Income_USD", "Daily_Commute_km", "Number_of_Cars_Owned",
           "Charging_Stations_Near_Home", "Charging_Stations_Near_Work",
           "Environmental_Concern_Level"] + CATS].copy()
X["Range_Anx_ord"] = train["Range_Anxiety_Level"].map(ORD)
assert X["Range_Anx_ord"].notna().all()
for c in CATS:
    X[c] = X[c].fillna(X[c].mode()[0]).astype("category")
FEATS = list(X.columns)
cv = StratifiedKFold(5, shuffle=True, random_state=42)
oof = np.zeros(len(y))
imp = np.zeros(len(FEATS))
for tri, val in cv.split(X, y):
    m = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.03, num_leaves=63,
                           feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                           lambda_l2=5, random_state=42, verbose=-1)
    m.fit(X.iloc[tri], y[tri])
    oof[val] = m.predict_proba(X.iloc[val])[:, 1]
    imp += m.feature_importances_
print(f"LGB-ord OOF: {roc_auc_score(y, oof):.4f} (native-cat LGB was 0.9418)")
rank = dict(sorted(zip(FEATS, imp / 5), key=lambda t: -t[1]))
for i, (k, v) in enumerate(rank.items()):
    print(i + 1, k, round(float(v), 1))
np.save("oof_lgb_ord.npy", oof)
