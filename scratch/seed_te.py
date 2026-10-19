"""Seed-mean of TE model: TE4096-s5 encoding once (deterministic), LGBM seeds 42/43/44.
Raw-mean OOF. OOF only.
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
BINS, SMOOTH, SEEDS = 4096, 5.0, (42, 43, 44)


def base_params(seed):
    return dict(n_estimators=3000, learning_rate=0.03, num_leaves=63,
                min_child_samples=40, subsample=0.9, subsample_freq=1,
                colsample_bytree=0.8, reg_lambda=1.0, max_bin=255,
                random_state=seed, verbose=-1)


def digit_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = {}
    for column in NUMERIC:
        scaled = np.rint(frame[column].to_numpy(dtype=np.float64) * 1e4).astype(np.int64)
        for position in range(-4, 4):
            result[f"{column}__digit_{position}"] = ((scaled // (10 ** (position + 4))) % 10).astype(np.int8)
    return pd.DataFrame(result, index=frame.index)


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
y = (train[TARGET] == "Yes").astype(np.int8).to_numpy()
levels = {c: pd.Index(pd.concat([train[c], test[c]]).dropna().unique()) for c in CATEGORICAL}
base = train[NUMERIC + CATEGORICAL].copy()
for column in CATEGORICAL:
    base[column] = pd.Categorical(base[column].fillna(levels[column][0]), categories=levels[column])
base = pd.concat([base, digit_frame(train)], axis=1)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
te_oof = np.zeros(len(train))
for tri, vai in cv.split(base, y):
    edges = np.quantile(train["Annual_Income_USD"].to_numpy()[tri], np.linspace(0, 1, BINS + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    b_tr = np.clip(np.digitize(train["Annual_Income_USD"].to_numpy()[tri], edges[1:-1]), 0, BINS - 1)
    b_va = np.clip(np.digitize(train["Annual_Income_USD"].to_numpy()[vai], edges[1:-1]), 0, BINS - 1)
    sums = np.bincount(b_tr, weights=y[tri].astype(float), minlength=BINS)
    cnts = np.bincount(b_tr, minlength=BINS)
    prior = y[tri].mean()
    means = (sums + SMOOTH * prior) / (cnts + SMOOTH)
    p = np.clip(means[b_va], 1e-6, 1 - 1e-6)
    te_oof[vai] = np.log(p / (1 - p))
X = base.copy()
X["te_income_4096_s5"] = te_oof

oof_seeds = np.zeros((len(SEEDS), len(train)))
for s, seed in enumerate(SEEDS):
    for tri, vai in cv.split(X, y):
        Xtr = X.iloc[tri].copy()
        Xva = X.iloc[vai].copy()
        edges = np.quantile(train["Annual_Income_USD"].to_numpy()[tri], np.linspace(0, 1, BINS + 1))
        edges[0], edges[-1] = -np.inf, np.inf
        b_all = np.clip(np.digitize(train["Annual_Income_USD"].to_numpy(), edges[1:-1]), 0, BINS - 1)
        b_tri = b_all[tri]
        sums = np.bincount(b_tri, weights=y[tri].astype(float), minlength=BINS)
        cnts = np.bincount(b_tri, minlength=BINS)
        prior = y[tri].mean()
        means = (sums + SMOOTH * prior) / (cnts + SMOOTH)
        p = np.clip(means[b_all], 1e-6, 1 - 1e-6)
        Xtr["te_income_4096_s5"] = np.log(p[tri] / (1 - p[tri]))
        Xva["te_income_4096_s5"] = np.log(p[vai] / (1 - p[vai]))
        model = lgb.LGBMClassifier(**base_params(seed))
        model.fit(Xtr, y[tri], eval_set=[(Xva, y[vai])], eval_metric="auc",
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        oof_seeds[s, vai] = model.predict_proba(Xva)[:, 1]
    print(f"seed {seed}: OOF {roc_auc_score(y, oof_seeds[s]):.7f}", flush=True)
mean_oof = oof_seeds.mean(axis=0)
print(f"TE seed-mean OOF: {roc_auc_score(y, mean_oof):.7f} (solo-TE 0.9444800)", flush=True)
np.save("scratch/oof_seed_te.npy", mean_oof)
