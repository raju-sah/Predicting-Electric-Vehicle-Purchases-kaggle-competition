"""Step 5: diverse stack CBM + LGB + XGB. Pairwise corr, LogReg meta, gate 0.9421."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb

TARGET = "Will_Buy_EV"
GATE = 0.9416 + 0.0005
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
for c in CATS:
    X[c] = X[c].fillna(X[c].mode()[0])
Xl = X.copy()
for c in CATS:
    Xl[c] = Xl[c].astype("category")
Xt = test[FEATS].copy()
for c in CATS:
    Xt[c] = Xt[c].fillna(X[c].mode()[0])
Xtl = Xt.copy()
for c in CATS:
    Xtl[c] = Xtl[c].astype("category")

oof_c = np.load("oof_cbm.npy")
oof_l = np.load("oof_lgb.npy")
pre = ColumnTransformer([
    ("n", SimpleImputer(strategy="median"), [f for f in FEATS if f not in CATS]),
    ("c", make_pipeline(SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore")), CATS),
])
cv = StratifiedKFold(5, shuffle=True, random_state=42)
oof_x = np.zeros(len(y))
for tri, val in cv.split(X, y):
    m = make_pipeline(pre, XGBClassifier(
        n_estimators=500, learning_rate=0.03, max_depth=6, subsample=0.8,
        colsample_bytree=0.6, reg_lambda=5, tree_method="hist", random_state=42))
    m.fit(X.iloc[tri], y[tri])
    oof_x[val] = m.predict_proba(X.iloc[val])[:, 1]
print(f"XGB OOF: {roc_auc_score(y, oof_x):.4f}", flush=True)
np.save("oof_xgb2.npy", oof_x)
C = np.corrcoef([oof_c, oof_l, oof_x])
print("pairwise corr CBM/LGB/XGB:\n", np.round(C, 4))

Z = np.column_stack([oof_c, oof_l, oof_x])
meta = LogisticRegression(C=1.0)
moof = np.zeros(len(y))
for tri, val in cv.split(Z, y):
    meta.fit(Z[tri], y[tri])
    moof[val] = meta.predict_proba(Z[val])[:, 1]
s = roc_auc_score(y, moof)
print(f"DIVERSE STACK OOF: {s:.4f} gate {GATE:.4f} -> {'PASS' if s > GATE else 'FAIL'}",
      flush=True)
if s > GATE:
    pc = CatBoostClassifier(iterations=1000, learning_rate=0.03, depth=6,
                            l2_leaf_reg=5, random_seed=42, verbose=False)
    pc.fit(X, y, cat_features=CATS)
    pl = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.03, num_leaves=63,
                            feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                            lambda_l2=5, random_state=42, verbose=-1)
    pl.fit(Xl, y)
    px = make_pipeline(pre, XGBClassifier(
        n_estimators=500, learning_rate=0.03, max_depth=6, subsample=0.8,
        colsample_bytree=0.6, reg_lambda=5, tree_method="hist", random_state=42))
    px.fit(X, y)
    Zte = np.column_stack([pc.predict_proba(Xt)[:, 1], pl.predict_proba(Xtl)[:, 1],
                           px.predict_proba(Xt)[:, 1]])
    meta.fit(Z, y)
    pd.DataFrame({"id": test["id"], TARGET: meta.predict_proba(Zte)[:, 1]}).to_csv(
        "subs/v06_divstack.csv", index=False)
    print("wrote subs/v06_divstack.csv")
