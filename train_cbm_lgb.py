"""Step 1: CatBoost (native cats) + LightGBM (category dtype), same 5-fold seed 42."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier, Pool
import lightgbm as lgb

TARGET = "Will_Buy_EV"
FEATS = ["Age", "Annual_Income_USD", "Daily_Commute_km", "Number_of_Cars_Owned",
         "Charging_Stations_Near_Home", "Charging_Stations_Near_Work",
         "Environmental_Concern_Level", "Gender", "City_Type", "Current_Car_Type",
         "Home_Charging_Possible", "Subsidy_Available", "Range_Anxiety_Level"]
CATS = ["Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible",
        "Subsidy_Available", "Range_Anxiety_Level"]
train = pd.read_csv("train.csv")
print("Range_Anxiety_Level unique:", train["Range_Anxiety_Level"].unique())
y = (train[TARGET] == "Yes").astype(int).to_numpy()
X = train[FEATS].copy()
for c in CATS:
    X[c] = X[c].fillna(X[c].mode()[0])
Xl = X.copy()
for c in CATS:
    Xl[c] = Xl[c].astype("category")

cv = StratifiedKFold(5, shuffle=True, random_state=42)
oof_c, oof_l = np.zeros(len(y)), np.zeros(len(y))
imp_c = np.zeros(len(FEATS))
impl = np.zeros(len(FEATS))
for i, (tri, val) in enumerate(cv.split(X, y)):
    cm = CatBoostClassifier(iterations=1000, learning_rate=0.03, depth=6,
                            l2_leaf_reg=5, random_seed=42, verbose=False)
    cm.fit(X.iloc[tri], y[tri], cat_features=CATS)
    oof_c[val] = cm.predict_proba(X.iloc[val])[:, 1]
    imp_c += cm.get_feature_importance(Pool(X.iloc[tri], y[tri], cat_features=CATS))
    print(f"cbm fold {i} AUC: {roc_auc_score(y[val], oof_c[val]):.4f}", flush=True)
    lm = lgb.LGBMClassifier(n_estimators=1000, learning_rate=0.03, num_leaves=63,
                            feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                            lambda_l2=5, random_state=42, verbose=-1)
    lm.fit(Xl.iloc[tri], y[tri])
    oof_l[val] = lm.predict_proba(Xl.iloc[val])[:, 1]
    impl += lm.feature_importances_
    print(f"lgb fold {i} AUC: {roc_auc_score(y[val], oof_l[val]):.4f}", flush=True)

print(f"CBM OOF: {roc_auc_score(y, oof_c):.4f}")
print(f"LGB OOF: {roc_auc_score(y, oof_l):.4f}")
print("CBM importance:", dict(sorted(zip(FEATS, imp_c / 5), key=lambda t: -t[1])))
print("LGB importance:", dict(sorted(zip(FEATS, impl / 5), key=lambda t: -t[1])))
np.save("oof_cbm.npy", oof_c)
np.save("oof_lgb.npy", oof_l)
