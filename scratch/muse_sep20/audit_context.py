"""MUSE-SEP20 context audit: verify v66-v68 OOF integrity + pool diversity."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

train = pd.read_csv("train.csv", usecols=["id", "Will_Buy_EV"])
y = (train["Will_Buy_EV"] == "Yes").to_numpy(np.int8)
fold = np.full(len(y), -1, np.int8)
for k, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(y, y)):
    fold[va] = k

mem = pd.read_csv("scratch/pull_sep18/memory_recipe_out/oof_predictions.csv")
print("memory fold col matches gate:", np.array_equal(mem["fold"].to_numpy(), fold))
print("memory labels match:", np.array_equal(mem["Will_Buy_EV"].to_numpy(), y))

for v in ["v66", "v67", "v68", "v69", "v70"]:
    o = np.load(f"scratch/verified_ensembles/oof_{v}.npy")
    per = [roc_auc_score(y[fold == k], o[fold == k]) for k in range(5)]
    print(f"{v}: pooled={roc_auc_score(y, o):.7f} mean={np.mean(per):.7f} "
          + " ".join(f"{a:.5f}" for a in per))

pool = {"memory": mem["oof_final"].to_numpy(float)}
for v in ["v66", "v67", "v68"]:
    pool[v] = np.load(f"scratch/verified_ensembles/oof_{v}.npy")
pool["digit"] = np.load("oof_digit_features.npy")
pool["lgb"] = np.load("oof_lgb.npy")
names = list(pool)
print("pairwise spearman (OOF):")
for i, a in enumerate(names):
    row = " ".join(f"{spearmanr(pool[a], pool[b])[0]:.5f}" for b in names)
    print(f"  {a:7s} {row}")
print("pooled AUC:", {n: round(float(roc_auc_score(y, pool[n])), 6) for n in names})
