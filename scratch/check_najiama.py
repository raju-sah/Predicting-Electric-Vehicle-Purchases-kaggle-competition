import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

y = (pd.read_csv("train.csv", usecols=["Will_Buy_EV"])["Will_Buy_EV"] == "Yes").astype(int).to_numpy()
v68 = np.load("scratch/verified_ensembles/oof_v68.npy")
print(f"v68 OOF={roc_auc_score(y, v68):.7f}")
import glob
for f in sorted(glob.glob("scratch/pull_sep16/ds_najiama/*.csv")):
    try:
        df = pd.read_csv(f)
    except Exception as e:  # noqa: BLE001
        print(f, "read-err", e)
        continue
    cols = [c for c in df.columns if c != "id"]
    print(f, df.shape, cols)
    for c in cols:
        v = pd.to_numeric(df[c], errors="coerce").to_numpy()
        if len(v) == len(y) and np.isfinite(v).all():
            r = np.corrcoef(np.argsort(np.argsort(v)), np.argsort(np.argsort(v68)))[0, 1]
            print(f"   {c}: OOF={roc_auc_score(y, v):.7f} rho-vs-v68={r:.5f}")
