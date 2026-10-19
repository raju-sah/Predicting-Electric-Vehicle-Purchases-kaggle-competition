"""V02: tuned HGB + XGB, same 5-fold, average blend."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

TARGET = "Will_Buy_EV"
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
y = (train[TARGET] == "Yes").astype(int)
X = train.drop(columns=[TARGET, "id"])

num = X.select_dtypes("number").columns.tolist()
cat = [c for c in X.columns if c not in num]
pre = ColumnTransformer([
    ("n", SimpleImputer(strategy="median"), num),
    ("c", make_pipeline(SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore")), cat),
])
cv = StratifiedKFold(5, shuffle=True, random_state=42)
hgb = make_pipeline(pre, HistGradientBoostingClassifier(
    learning_rate=0.05, max_iter=600, l2_regularization=5, random_state=42))
xgb = make_pipeline(pre, XGBClassifier(
    n_estimators=500, learning_rate=0.03, max_depth=6, subsample=0.8,
    colsample_bytree=0.6, reg_lambda=5, tree_method="hist", random_state=42))

oof_h = cross_val_predict(hgb, X, y, cv=cv, method="predict_proba")[:, 1]
print(f"HGB OOF: {roc_auc_score(y, oof_h):.4f}", flush=True)
oof_x = cross_val_predict(xgb, X, y, cv=cv, method="predict_proba")[:, 1]
print(f"XGB OOF: {roc_auc_score(y, oof_x):.4f}", flush=True)
blend = (oof_h + oof_x) / 2
print(f"BLEND OOF: {roc_auc_score(y, blend):.4f}", flush=True)

hgb.fit(X, y)
xgb.fit(X, y)
p = (hgb.predict_proba(test[X.columns])[:, 1] + xgb.predict_proba(test[X.columns])[:, 1]) / 2
pd.DataFrame({"id": test["id"], TARGET: p}).to_csv("subs/v02_hgb_xgb.csv", index=False)
print("wrote subs/v02_hgb_xgb.csv")
