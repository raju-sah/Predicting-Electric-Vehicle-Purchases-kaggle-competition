"""S6E9 baseline: HistGradientBoosting, 5-fold OOF ROC-AUC. No new deps."""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

TARGET = "Will_Buy_EV"
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
y = (train[TARGET] == "Yes").astype(int)
X = train.drop(columns=[TARGET, "id"])

num = X.select_dtypes("number").columns.tolist()
cat = [c for c in X.columns if c not in num]
print("num:", num, "cat:", cat, "pos_rate:", round(y.mean(), 4))
pre = ColumnTransformer([
    ("n", SimpleImputer(strategy="median"), num),
    ("c", make_pipeline(SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore")), cat),
])
clf = make_pipeline(pre, HistGradientBoostingClassifier(random_state=42))
oof = cross_val_predict(clf, X, y, cv=StratifiedKFold(5, shuffle=True, random_state=42),
                        method="predict_proba")[:, 1]
print(f"OOF AUC: {roc_auc_score(y, oof):.4f}")
clf.fit(X, y)
sub = pd.DataFrame({"id": test["id"], TARGET: clf.predict_proba(test[X.columns])[:, 1]})
sub.to_csv("subs/v01_hgb_base.csv", index=False)
print("wrote subs/v01_hgb_base.csv", sub.shape)
