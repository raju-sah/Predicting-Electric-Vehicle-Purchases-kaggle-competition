"""V04: (HGB + XGB42 + XGB43 + XGB44)/4 using saved v02 = (HGB+XGB42)/2."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
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
v02 = pd.read_csv("subs/v02_hgb_xgb.csv")
assert (v02["id"] == test["id"]).all()
blend = v02[TARGET].to_numpy() * 2  # = HGB + XGB42
for s in (43, 44):
    clf = make_pipeline(pre, XGBClassifier(
        n_estimators=500, learning_rate=0.03, max_depth=6, subsample=0.8,
        colsample_bytree=0.6, reg_lambda=5, tree_method="hist", random_state=s))
    clf.fit(X, y)
    blend += clf.predict_proba(test[X.columns])[:, 1]
    print(f"seed {s} fitted", flush=True)
pd.DataFrame({"id": test["id"], TARGET: blend / 4}).to_csv("subs/v04_4way.csv", index=False)
print("wrote subs/v04_4way.csv")
