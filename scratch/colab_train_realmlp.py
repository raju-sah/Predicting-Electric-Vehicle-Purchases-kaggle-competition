# RealMLP TD_CLASSIF on GPU (Colab T4) — new decorrelated member for the 0.94651 base.
# Exact gate folds: StratifiedKFold(5, shuffle=True, random_state=42).
# Single seed first (5 fits); OOF + test preds saved to /content/.
import numpy as np, pandas as pd, time
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import torch
from pytabkit import RealMLP_TD_Classifier

t0 = time.time()
TARGET, ID = "Will_Buy_EV", "id"
CATS = ["Gender", "Home_Charging_Possible", "Subsidy_Available", "Range_Anxiety_Level", "City_Type", "Current_Car_Type"]
train = pd.read_csv("/content/train.csv")
test = pd.read_csv("/content/test.csv")
y = (train[TARGET] == "Yes").astype(np.int64).to_numpy()

feats = [c for c in train.columns if c not in (ID, TARGET)]
X = train[feats].copy()
Xt = test[feats].copy()
for c in CATS:
    X[c] = X[c].astype("category")
    Xt[c] = Xt[c].astype(pd.api.types.CategoricalDtype(categories=X[c].cat.categories))
for c in X.columns:
    if c not in CATS:
        X[c] = X[c].astype(np.float32)
        Xt[c] = Xt[c].astype(np.float32)
print("X:", X.shape, "| dtypes ok", flush=True)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(X))
test_pred = np.zeros(len(Xt))
fold_aucs = []
for f, (tr, va) in enumerate(skf.split(X, y)):
    tf = time.time()
    try:
        model = RealMLP_TD_Classifier(random_state=42 + f, device="cuda")
    except TypeError:
        model = RealMLP_TD_Classifier(random_state=42 + f)
    model.fit(X.iloc[tr], y[tr])
    p = model.predict_proba(X.iloc[va])[:, 1]
    oof[va] = p
    test_pred += model.predict_proba(Xt)[:, 1] / 5
    auc = roc_auc_score(y[va], p)
    fold_aucs.append(auc)
    print(f"fold {f}: auc {auc:.6f} | {time.time()-tf:.0f}s | total {time.time()-t0:.0f}s", flush=True)

oof_auc = roc_auc_score(y, oof)
print(f"\nOOF AUC: {oof_auc:.6f} | folds: {[round(a,6) for a in fold_aucs]}", flush=True)
np.save("/content/oof_realmlp_gpu.npy", oof)
np.save("/content/test_realmlp_gpu.npy", test_pred)
pd.DataFrame({ID: test[ID].to_numpy(), "pred": test_pred}).to_csv("/content/test_realmlp_gpu.csv", index=False)
pd.DataFrame({ID: train[ID].to_numpy(), "oof": oof}).to_csv("/content/oof_realmlp_gpu.csv", index=False)
print("SAVED /content/oof_realmlp_gpu.csv /content/test_realmlp_gpu.csv", flush=True)
