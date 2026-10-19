"""V03: smoothed OOF target-encoding on 6 cat cols + 3-seed HGB bag on num+TE."""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

TARGET = "Will_Buy_EV"
ALPHA, SEEDS = 50, (42, 43, 44)
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
y = ((train[TARGET] == "Yes").astype(int)).to_numpy()
num_cols = ["Age", "Annual_Income_USD", "Daily_Commute_km", "Number_of_Cars_Owned",
            "Charging_Stations_Near_Home", "Charging_Stations_Near_Work",
            "Environmental_Concern_Level"]
cat_cols = ["Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible",
            "Subsidy_Available", "Range_Anxiety_Level"]
Xtr_raw, Xte_raw = train[num_cols + cat_cols].copy(), test[num_cols + cat_cols].copy()
for c in cat_cols:  # ponytail: most_frequent fill, no extra category
    m = Xtr_raw[c].mode()[0]
    Xtr_raw[c] = Xtr_raw[c].fillna(m)
    Xte_raw[c] = Xte_raw[c].fillna(m)

med = Xtr_raw[num_cols].median()
Xn_tr = SimpleImputer(strategy="median").fit_transform(Xtr_raw[num_cols])
Xn_te = SimpleImputer(strategy="median").fit_transform(Xte_raw[num_cols])
glob = y.mean()
cv = StratifiedKFold(5, shuffle=True, random_state=42)
te_oof = np.zeros((len(y), len(cat_cols)))
te_tst = np.zeros((len(Xte_raw), len(cat_cols)))
for tri, val in cv.split(Xn_tr, y):
    for j, c in enumerate(cat_cols):
        g = pd.Series(y[tri]).groupby(Xtr_raw[c].iloc[tri].to_numpy()).agg(["sum", "count"])
        enc = (g["sum"] + ALPHA * glob) / (g["count"] + ALPHA)
        te_oof[val, j] = Xtr_raw[c].iloc[val].map(enc).fillna(glob).to_numpy()
        te_tst[:, j] += Xte_raw[c].map(enc).fillna(glob).to_numpy() / 5
F_tr = np.hstack([Xn_tr, te_oof])
F_te = np.hstack([Xn_te, te_tst])

oof, pred = np.zeros(len(y)), np.zeros(len(Xte_raw))
for s in SEEDS:
    clf = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=600,
                                         l2_regularization=5, random_state=s)
    o = np.zeros(len(y))
    for tri, val in cv.split(F_tr, y):
        clf.fit(F_tr[tri], y[tri])
        o[val] = clf.predict_proba(F_tr[val])[:, 1]
    print(f"seed {s} OOF: {roc_auc_score(y, o):.4f}", flush=True)
    oof += o / len(SEEDS)
    clf.fit(F_tr, y)
    pred += clf.predict_proba(F_te)[:, 1] / len(SEEDS)
print(f"BAG OOF: {roc_auc_score(y, oof):.4f}", flush=True)
pd.DataFrame({"id": test["id"], TARGET: pred}).to_csv("subs/v03_te_bag.csv", index=False)
print("wrote subs/v03_te_bag.csv")
