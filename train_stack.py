"""V05: LogReg stack over HGB-tuned + XGB + HGB-TE OOFs."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

TARGET = "Will_Buy_EV"
ALPHA = 50
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
y = ((train[TARGET] == "Yes").astype(int)).to_numpy()
X = train.drop(columns=[TARGET, "id"])
num = X.select_dtypes("number").columns.tolist()
cat = [c for c in X.columns if c not in num]
pre = ColumnTransformer([
    ("n", SimpleImputer(strategy="median"), num),
    ("c", make_pipeline(SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore")), cat),
])
Xt = pre.fit_transform(X)
Xte = pre.transform(test[X.columns])

# TE features (OOF-style for train via same folds, full-fit map for test)
te_cols = cat
cv = StratifiedKFold(5, shuffle=True, random_state=42)
glob = y.mean()
TE_tr = np.zeros((len(y), len(te_cols)))
TE_te = np.zeros((len(test), len(te_cols)))
for tri, val in cv.split(X, y):
    for j, c in enumerate(te_cols):
        g = pd.Series(y[tri]).groupby(X[c].iloc[tri].to_numpy()).agg(["sum", "count"])
        enc = (g["sum"] + ALPHA * glob) / (g["count"] + ALPHA)
        TE_tr[val, j] = X[c].iloc[val].map(enc).fillna(glob).to_numpy()
        TE_te[:, j] += test[c].map(enc).fillna(glob).to_numpy() / 5
F_tr = np.hstack([Xt.toarray() if hasattr(Xt, "toarray") else Xt, TE_tr])
F_te = np.hstack([Xte.toarray() if hasattr(Xte, "toarray") else Xte, TE_te])
_ = F_tr  # TE used only by hgb_te below via rebuilt matrices

bases = {
    "hgb": make_pipeline(pre, HistGradientBoostingClassifier(
        learning_rate=0.05, max_iter=600, l2_regularization=5, random_state=42)),
    "xgb": make_pipeline(pre, XGBClassifier(
        n_estimators=500, learning_rate=0.03, max_depth=6, subsample=0.8,
        colsample_bytree=0.6, reg_lambda=5, tree_method="hist", random_state=42)),
}
OOF, TESTP = {}, {}
for name, pipe in bases.items():
    o = np.zeros(len(y))
    for tri, val in cv.split(X, y):
        pipe.fit(X.iloc[tri], y[tri])
        o[val] = pipe.predict_proba(X.iloc[val])[:, 1]
    OOF[name] = o
    print(f"{name} OOF: {roc_auc_score(y, o):.4f}", flush=True)
    pipe.fit(X, y)
    TESTP[name] = pipe.predict_proba(test[X.columns])[:, 1]

# HGB on num+TE (dense small)
Xn_all = np.hstack([SimpleImputer(strategy="median").fit_transform(train[num]),
                    TE_tr])
Xn_te = np.hstack([SimpleImputer(strategy="median").fit_transform(test[num]),
                   TE_te])
o = np.zeros(len(y))
for tri, val in cv.split(Xn_all, y):
    m = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=600,
                                       l2_regularization=5, random_state=42)
    m.fit(Xn_all[tri], y[tri])
    o[val] = m.predict_proba(Xn_all[val])[:, 1]
OOF["hte"] = o
print(f"hte OOF: {roc_auc_score(y, o):.4f}", flush=True)
m = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=600,
                                   l2_regularization=5, random_state=42)
m.fit(Xn_all, y)
TESTP["hte"] = m.predict_proba(Xn_te)[:, 1]

Z_tr = np.column_stack([OOF["hgb"], OOF["xgb"], OOF["hte"]])
Z_te = np.column_stack([TESTP["hgb"], TESTP["xgb"], TESTP["hte"]])
meta = LogisticRegression(C=1.0)
moof = np.zeros(len(y))
for tri, val in cv.split(Z_tr, y):
    meta.fit(Z_tr[tri], y[tri])
    moof[val] = meta.predict_proba(Z_tr[val])[:, 1]
print(f"STACK OOF: {roc_auc_score(y, moof):.4f}", flush=True)
print("avg base OOF:", round(np.mean([roc_auc_score(y, OOF[k]) for k in OOF]), 4))
meta.fit(Z_tr, y)
final = meta.predict_proba(Z_te)[:, 1]
pd.DataFrame({"id": test["id"], TARGET: final}).to_csv("subs/v05_stack.csv", index=False)
print("wrote subs/v05_stack.csv")
