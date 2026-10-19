"""Final subs: full-fit LGB solo + weighted blend CBM/LGB/XGB/HGB (OOF-weighted)."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb

TARGET = "Will_Buy_EV"
FEATS = ["Age", "Annual_Income_USD", "Daily_Commute_km", "Number_of_Cars_Owned",
         "Charging_Stations_Near_Home", "Charging_Stations_Near_Work",
         "Environmental_Concern_Level", "Gender", "City_Type", "Current_Car_Type",
         "Home_Charging_Possible", "Subsidy_Available", "Range_Anxiety_Level"]
CATS = ["Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible",
        "Subsidy_Available", "Range_Anxiety_Level"]
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
y = (train[TARGET] == "Yes").astype(int).to_numpy()
X = train[FEATS].copy()
Xt = test[FEATS].copy()
for c in CATS:
    m = X[c].mode()[0]
    X[c] = X[c].fillna(m)
    Xt[c] = Xt[c].fillna(m)
Xl, Xtl = X.copy(), Xt.copy()
for c in CATS:
    Xl[c] = Xl[c].astype("category")
    Xtl[c] = Xtl[c].astype("category")

pl = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.03, num_leaves=63,
                        feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                        lambda_l2=5, random_state=42, verbose=-1)
pl.fit(Xl, y)
p_lgb = pl.predict_proba(Xtl)[:, 1]
print("lgb fitted", flush=True)
pd.DataFrame({"id": test["id"], TARGET: p_lgb}).to_csv("subs/v06_lgb.csv", index=False)

pc = CatBoostClassifier(iterations=1000, learning_rate=0.03, depth=6,
                        l2_leaf_reg=5, random_seed=42, verbose=False)
pc.fit(X, y, cat_features=CATS)
p_cbm = pc.predict_proba(Xt)[:, 1]
print("cbm fitted", flush=True)
pre = ColumnTransformer([
    ("n", SimpleImputer(strategy="median"), [f for f in FEATS if f not in CATS]),
    ("c", make_pipeline(SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore")), CATS),
])
px = make_pipeline(pre, XGBClassifier(
    n_estimators=500, learning_rate=0.03, max_depth=6, subsample=0.8,
    colsample_bytree=0.6, reg_lambda=5, tree_method="hist", random_state=42))
px.fit(X, y)
p_xgb = px.predict_proba(Xt)[:, 1]
print("xgb fitted", flush=True)
ph = make_pipeline(pre, HistGradientBoostingClassifier(
    learning_rate=0.05, max_iter=600, l2_regularization=5, random_state=42))
ph.fit(X, y)
p_hgb = ph.predict_proba(Xt)[:, 1]
print("hgb fitted", flush=True)

w = np.array([0.9414, 0.9418, 0.9417, 0.9414])  # OOF weights
w = w / w.sum()
blend = w[0] * p_cbm + w[1] * p_lgb + w[2] * p_xgb + w[3] * p_hgb
pd.DataFrame({"id": test["id"], TARGET: blend}).to_csv("subs/v07_blend4.csv", index=False)
print("wrote v06 + v07")
