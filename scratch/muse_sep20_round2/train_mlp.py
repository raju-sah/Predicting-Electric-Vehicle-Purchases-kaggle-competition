"""Round2 model F: sklearn MLP on digit frame (NEW learner family locally).

Neural nets are the only major family never trained in this project (RealMLP
exists only as external OOF). Proof-of-signal with cheap sklearn MLP first;
escalate to PyTorch embedding-MLP only if decorrelated + reasonable OOF.
Per-fold StandardScaler fit on fit_idx only (leak-free). Exact gate folds.
Outputs: scratch/muse_sep20_round2/oof_mlp.npy, test_mlp.npy, run_mlp.json
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

T0 = time.time()
TARGET = "Will_Buy_EV"
OUT = "scratch/muse_sep20_round2"
NUMERIC = [
    "Age", "Annual_Income_USD", "Daily_Commute_km", "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home", "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
]
CATS = ["Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible",
        "Subsidy_Available", "Range_Anxiety_Level"]
DIGIT_POSITIONS = range(-4, 4)


def digit_frame(frame: pd.DataFrame) -> pd.DataFrame:
    result = {}
    for column in NUMERIC:
        values = frame[column].to_numpy(dtype=np.float64)
        for position in DIGIT_POSITIONS:
            result[f"{column}__d{position}"] = np.mod(
                np.floor(values / (10.0 ** position)), 10.0).astype(np.float64)
    return pd.DataFrame(result, index=frame.index)


train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
y = (train[TARGET] == "Yes").astype(np.int8).to_numpy()
levels = {c: sorted(pd.concat([train[c], test[c]]).dropna().unique()) for c in CATS}


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    num = frame[NUMERIC].copy()
    dig = digit_frame(frame)
    cats = pd.DataFrame(
        {c: pd.Categorical(frame[c].fillna(levels[c][0]), categories=levels[c]).codes
         for c in CATS}, index=frame.index).astype(np.float64)
    flags = pd.DataFrame({
        "f_spike": (np.rint(frame["Annual_Income_USD"].to_numpy(float)) == 30000).astype(float),
        "f_cliff": (frame["Annual_Income_USD"].to_numpy(float) >= 170537).astype(float),
        "f_dead": ((frame["Annual_Income_USD"].to_numpy(float) >= 38000)
                   & (frame["Annual_Income_USD"].to_numpy(float) <= 42000)).astype(float),
    }, index=frame.index)
    return pd.concat([num.astype(float), dig, cats, flags], axis=1)


X = prepare(train).to_numpy(np.float32)
Xt = prepare(test).to_numpy(np.float32)
print(f"features {X.shape[1]}", flush=True)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(y), np.float64)
test_pred = np.zeros(len(test), np.float64)
fold_auc = []
for fold, (fit_idx, valid_idx) in enumerate(cv.split(X, y)):
    sc = StandardScaler().fit(X[fit_idx])
    model = MLPClassifier(hidden_layer_sizes=(128, 64), activation="relu",
                          alpha=0.001, learning_rate_init=0.001,
                          max_iter=150, early_stopping=True, n_iter_no_change=10,
                          validation_fraction=0.1, random_state=42 + fold,
                          verbose=False)
    model.fit(sc.transform(X[fit_idx]), y[fit_idx])
    oof[valid_idx] = model.predict_proba(sc.transform(X[valid_idx]))[:, 1]
    test_pred += model.predict_proba(sc.transform(Xt))[:, 1] / cv.n_splits
    auc = roc_auc_score(y[valid_idx], oof[valid_idx])
    fold_auc.append(float(auc))
    print(f"fold {fold}: AUC {auc:.7f} iters {model.n_iter_} "
          f"elapsed {(time.time()-T0)/60:.1f}min", flush=True)

pooled = float(roc_auc_score(y, oof))
np.save(f"{OUT}/oof_mlp.npy", oof)
np.save(f"{OUT}/test_mlp.npy", test_pred)
json.dump({"pooled_oof_auc": pooled, "fold_auc": fold_auc,
           "mean_fold_auc": float(np.mean(fold_auc)),
           "folds": "StratifiedKFold(5, shuffle=True, random_state=42)",
           "mechanism": "sklearn MLP(128,64) on scaled digit frame (proof of signal)"},
          open(f"{OUT}/run_mlp.json", "w"), indent=2)
print(f"MLP DONE pooled={pooled:.7f}", flush=True)
