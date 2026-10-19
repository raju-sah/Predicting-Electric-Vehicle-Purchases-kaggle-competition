"""Cross-family rank blend from frozen OOF+test pairs. Ranks ACROSS families.
Weights fixed a priori (3 candidates). Selection among 3 on OOF reuses labels — disclosed.
Members: v68 logit-ens (GBDT), catA (CatBoost), hgbB (HGB), digit (LGBM-digit).
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

y = (pd.read_csv("train.csv", usecols=["Will_Buy_EV"])["Will_Buy_EV"] == "Yes").astype(int).to_numpy()
sample = pd.read_csv("sample_submission.csv")

oof = {
    "v68": np.load("scratch/verified_ensembles/oof_v68.npy"),
    "catA": np.load("scratch/muse_sep20/oof_catA.npy"),
    "hgbB": np.load("scratch/muse_sep20/oof_hgbB.npy"),
    "digit": np.load("oof_digit_features.npy"),
}
v68t = pd.read_csv("subs/v68_verified5_logit.csv").set_index("id")["Will_Buy_EV"]
catAt = np.load("scratch/muse_sep20/test_catA.npy")
hgbBt = np.load("scratch/muse_sep20/test_hgbB.npy")
digit_df = pd.read_csv("subs/v08_digit_features.csv").set_index("id")
test_ids = pd.read_csv("test.csv", usecols=["id"])["id"].to_numpy()
tst = {
    "v68": v68t.reindex(sample["id"]).to_numpy(dtype=float),
    "catA": pd.Series(catAt, index=test_ids).reindex(sample["id"]).to_numpy(dtype=float),
    "hgbB": pd.Series(hgbBt, index=test_ids).reindex(sample["id"]).to_numpy(dtype=float),
    "digit": digit_df.reindex(sample["id"])["Will_Buy_EV"].to_numpy(dtype=float),
}

for m, v in oof.items():
    print(f"{m}: OOF={roc_auc_score(y, v):.7f}", flush=True)
names = list(oof)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        r = np.corrcoef(np.argsort(np.argsort(oof[names[i]])), np.argsort(np.argsort(oof[names[j]])))[0, 1]
        print(f"rho {names[i]}-{names[j]}: {r:.5f}", flush=True)

R_oof = {m: rankdata(v, method="average") / len(v) for m, v in oof.items()}
R_tst = {m: rankdata(v, method="average") / len(v) for m, v in tst.items()}
cands = {
    "xfam_equal": {m: 0.25 for m in names},
    "xfam_v68heavy": {"v68": 0.5, "catA": 0.2, "hgbB": 0.15, "digit": 0.15},
    "xfam_top2": {"v68": 0.5, "catA": 0.25, "hgbB": 0.25, "digit": 0.0},
}
for cn, w in cands.items():
    bo = sum(R_oof[m] * w[m] for m in names)
    auc = roc_auc_score(y, bo)
    print(f"{cn}: OOF={auc:.7f} {'GATE-PASS' if auc > 0.9462709 + 0.0005 else 'below gate'}", flush=True)
    bt = sum(R_tst[m] * w[m] for m in names)
    pd.DataFrame({"id": sample["id"], "Will_Buy_EV": bt}).to_csv(f"subs/{cn}.csv", index=False)
print("wrote subs/xfam_*.csv (submit ONLY on gate pass)", flush=True)
