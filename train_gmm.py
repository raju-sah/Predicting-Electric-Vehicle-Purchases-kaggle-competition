"""GMM modes: BGM(10) per numeric col on train+test, mode id as extra cat, LGB 5-fold."""
import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

TARGET = "Will_Buy_EV"
NUM = ["Age", "Annual_Income_USD", "Daily_Commute_km", "Number_of_Cars_Owned",
       "Charging_Stations_Near_Home", "Charging_Stations_Near_Work",
       "Environmental_Concern_Level"]
CATS = ["Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible",
        "Subsidy_Available", "Range_Anxiety_Level"]
tr = pd.read_csv("train.csv")
te = pd.read_csv("test.csv")
y = (tr[TARGET] == "Yes").astype(int).to_numpy()
med = tr[NUM].median()
TRn = tr[NUM].fillna(med).to_numpy()
TEn = te[NUM].fillna(med).to_numpy()
X = tr[CATS].copy()
Xt = te[CATS].copy()
for c in CATS:
    m = X[c].mode()[0]
    X[c] = X[c].fillna(m).astype("category")
    Xt[c] = Xt[c].fillna(m).astype("category")
for j, c in enumerate(NUM):
    g = BayesianGaussianMixture(n_components=10, random_state=42).fit(
        np.concatenate([TRn[:, [j]], TEn[:, [j]]]))
    X[f"mode_{c}"] = pd.Categorical(g.predict(TRn[:, [j]]))
    Xt[f"mode_{c}"] = pd.Categorical(g.predict(TEn[:, [j]]))
    print(c, "active modes:", len(np.unique(g.predict(TRn[:, [j]]))), flush=True)
P = dict(n_estimators=1000, learning_rate=0.03, num_leaves=63, feature_fraction=0.8,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=5, random_state=42, verbose=-1)
cv = StratifiedKFold(5, shuffle=True, random_state=42)
oof = np.zeros(len(y))
for tri, val in cv.split(X, y):
    m = lgb.LGBMClassifier(**P)
    m.fit(X.iloc[tri], y[tri])
    oof[val] = m.predict_proba(X.iloc[val])[:, 1]
print(f"LGB+GMMmodes OOF: {roc_auc_score(y, oof):.4f} (base 0.9418)", flush=True)
