"""LightGBM with unsupervised numeric codebooks for reported discrete values.

The source columns have finite support (for example, 805 commute readings and
13k income readings) and visibly non-smooth conditional rates.  This retains
the native numeric columns, while adding target-free categorical codebooks:
every discrete count/value and fixed reporting-resolution bins for income and
commute.  It is distinct from target encoding, GMM modes, and manual feature
interactions.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

TARGET = "Will_Buy_EV"
NUMERIC = [
    "Age", "Annual_Income_USD", "Daily_Commute_km", "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home", "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
]
CATEGORICAL = [
    "Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible",
    "Subsidy_Available", "Range_Anxiety_Level",
]
PARAMS = dict(
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=63,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=5,
    random_state=42,
    verbose=-1,
)


def add_codebooks(train_frame: pd.DataFrame, test_frame: pd.DataFrame):
    """Add train/test-consistent categorical codebooks without target labels."""
    train_out = train_frame[NUMERIC + CATEGORICAL].copy()
    test_out = test_frame[NUMERIC + CATEGORICAL].copy()
    for column in CATEGORICAL:
        mode = train_out[column].mode()[0]
        train_out[column] = train_out[column].fillna(mode).astype("category")
        test_out[column] = test_out[column].fillna(mode).astype("category")

    # Exact categories are appropriate for fields whose reported support is
    # already intrinsically discrete.  For the two longer supports, use their
    # stated reporting resolutions rather than target-derived bins.
    code_specs = {
        "Age_code": train_frame["Age"].astype(int),
        "Cars_code": train_frame["Number_of_Cars_Owned"].astype(int),
        "Home_stations_code": train_frame["Charging_Stations_Near_Home"].astype(int),
        "Work_stations_code": train_frame["Charging_Stations_Near_Work"].astype(int),
        "Concern_code": train_frame["Environmental_Concern_Level"].astype(int),
        "Income_thousands": np.floor(train_frame["Annual_Income_USD"] / 1000).astype(int),
        "Commute_km_code": np.floor(train_frame["Daily_Commute_km"]).astype(int),
    }
    test_specs = {
        "Age_code": test_frame["Age"].astype(int),
        "Cars_code": test_frame["Number_of_Cars_Owned"].astype(int),
        "Home_stations_code": test_frame["Charging_Stations_Near_Home"].astype(int),
        "Work_stations_code": test_frame["Charging_Stations_Near_Work"].astype(int),
        "Concern_code": test_frame["Environmental_Concern_Level"].astype(int),
        "Income_thousands": np.floor(test_frame["Annual_Income_USD"] / 1000).astype(int),
        "Commute_km_code": np.floor(test_frame["Daily_Commute_km"]).astype(int),
    }
    for name, values in code_specs.items():
        categories = np.union1d(values.unique(), test_specs[name].unique())
        train_out[name] = pd.Categorical(values, categories=categories)
        test_out[name] = pd.Categorical(test_specs[name], categories=categories)
    return train_out, test_out


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
y = (train[TARGET] == "Yes").astype(np.int8).to_numpy()
X, X_test = add_codebooks(train, test)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(train), dtype=np.float64)

for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y), start=1):
    model = lgb.LGBMClassifier(**PARAMS)
    model.fit(X.iloc[train_idx], y[train_idx])
    oof[valid_idx] = model.predict_proba(X.iloc[valid_idx])[:, 1]
    print(f"fold {fold}: OOF AUC {roc_auc_score(y[valid_idx], oof[valid_idx]):.7f}", flush=True)

auc = roc_auc_score(y, oof)
print(f"numeric-codebook OOF AUC: {auc:.7f}", flush=True)
np.save("oof_numeric_codebook.npy", oof)

# Train once on all labeled competition rows to make a deterministic probe
# submission file. Submission still requires the documented OOF gate.
model = lgb.LGBMClassifier(**PARAMS)
model.fit(X, y)
pred = model.predict_proba(X_test)[:, 1]
pd.DataFrame({"id": test["id"], TARGET: pred}).to_csv(
    "subs/numeric_codebook_probe.csv", index=False
)
print("wrote oof_numeric_codebook.npy and subs/numeric_codebook_probe.csv")
