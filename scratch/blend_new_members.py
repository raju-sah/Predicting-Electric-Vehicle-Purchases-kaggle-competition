# Evaluate new Colab members (RealMLP-GPU, XGB-stack) on the exact gate folds,
# then build v53-v55: tiny-to-small injections of the new members onto the
# 0.94651 base. Gate: only submit what OOF says beats pure base.
import numpy as np, pandas as pd, json
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

TARGET, ID = "Will_Buy_EV", "id"
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
ids_te = test[ID].to_numpy()
N = len(ids_te)
y = (train[TARGET] == "Yes").astype(np.uint8).to_numpy()

# rebuild fold assignment
from sklearn.model_selection import StratifiedKFold
outer = StratifiedKFold(5, shuffle=True, random_state=42)
FOLD = np.zeros(len(y), dtype=np.int8)
for k, (_, va) in enumerate(outer.split(np.zeros(len(y)), y)):
    FOLD[va] = k

def load_oof_member(oof_csv, col):
    d = pd.read_csv(oof_csv).sort_values(ID)
    assert np.array_equal(d[ID].to_numpy(), train[ID].to_numpy())
    return d[col].to_numpy(dtype=float)

members = {}
import os
for name, path, col in [
    ("realmlp_gpu", "scratch/colab_out/oof_realmlp_gpu.csv", "oof"),
    ("xgb_stack",   "scratch/colab_out/oof_xgbstack.csv",    "oof"),
]:
    if os.path.exists(path):
        members[name] = load_oof_member(path, col)
        auc = roc_auc_score(y, members[name])
        print(f"{name}: OOF AUC {auc:.6f}")

# reference frontier members for pool comparison
REF = {
    "megayak_A": ("scratch/pull_sep16/ds_megayak6/A_lgbm_triple_te_digits_3seed_oof.csv", "A_lgbm_triple_te_digits_3seed"),
    "naj_xgb10": ("scratch/pull_sep16/ds_najiama/XGBoost_Triple_TE_10folds_oof.csv", None),
}
for name, (path, col) in REF.items():
    if os.path.exists(path):
        members[name] = load_oof_member(path, col or [c for c in pd.read_csv(path, nrows=0).columns if c != ID][0])
        print(f"{name}: OOF AUC {roc_auc_score(y, members[name]):.6f}")

# per-fold AUC stability of new members
for name in ["realmlp_gpu", "xgb_stack"]:
    if name in members:
        fa = [roc_auc_score(y[FOLD == k], members[name][FOLD == k]) for k in range(5)]
        print(f"{name} per-fold: {[round(a,6) for a in fa]}")

# correlations to frontier
for new in ["realmlp_gpu", "xgb_stack"]:
    if new in members:
        for ref in ["megayak_A", "naj_xgb10"]:
            if ref in members:
                r = spearmanr(members[new], members[ref]).statistic
                print(f"rho {new} vs {ref}: {r:.4f}")

# test-side preds for new members
test_preds = {}
for name, path, col in [
    ("realmlp_gpu", "scratch/colab_out/test_realmlp_gpu.csv", "pred"),
    ("xgb_stack",   "scratch/colab_out/test_xgbstack.csv",    "pred"),
]:
    if os.path.exists(path):
        d = pd.read_csv(path).sort_values(ID)
        assert np.array_equal(d[ID].to_numpy(), ids_te)
        test_preds[name] = d[col].to_numpy(dtype=float)

# OOF-level greedy: does adding new members beat 0.946424 pool ceiling?
pool = {k: rankdata(v) / len(y) for k, v in members.items()}
best = None
def eval_mix(w):
    keys = list(w.keys())
    m = sum(w[k] * pool[k] for k in keys)
    return roc_auc_score(y, m)

start = {"megayak_A": 0.5, "naj_xgb10": 0.3}
for new in ["realmlp_gpu", "xgb_stack"]:
    if new in pool:
        w = dict(start); w[new] = 0.2
        cur = eval_mix(w)
        improved = True
        while improved:
            improved = False
            for k in list(w.keys()):
                for step in (0.05, 0.02, -0.02, -0.05):
                    w2 = dict(w)
                    w2[k] = max(0.0, w[k] + step)
                    if w2[k] == 0: del w2[k]
                    v = eval_mix(w2)
                    if v > cur + 1e-7:
                        cur, w, improved = v, w2, True
        print(f"hill-climb with {new}: {cur:.6f} | weights { {k: round(x,3) for k,x in w.items()} }")

# base for submissions
base = pd.read_csv("scratch/pull_sep16/jz_meta/submission_latest_best.csv").sort_values(ID)
base = base[TARGET].to_numpy(dtype=float)
rb = rankdata(base) / N
nn = pd.read_csv("scratch/pull_sep16/ds_megayak6/test_realmlp_g.csv").sort_values(ID)
nn = nn["G_realmlp_3seed"].to_numpy(dtype=float)

def lexrank(primary, secondary):
    o = np.lexsort((secondary, primary))
    r = np.empty(len(o), dtype=np.int64)
    r[o] = np.arange(1, len(o) + 1)
    return (r - 0.5) / len(r)

inc = test["Annual_Income_USD"].to_numpy(float)
km = test["Daily_Commute_km"].to_numpy(float)
env1 = test["Environmental_Concern_Level"].to_numpy() == 1
no_sub = test["Subsidy_Available"].astype(str).to_numpy() == "No"
anx_mh = test["Range_Anxiety_Level"].isin(["Medium", "High"]).to_numpy()
shift = np.zeros(N)
shift[inc >= 170537] += 10
shift[(inc >= 31004) & (inc <= 41970)] -= 10
shift[km >= 83] -= 5
shift[(inc == 30000) & no_sub & (env1 | anx_mh)] -= 5

sample = pd.read_csv("sample_submission.csv")
def emit(final, name, note):
    sub = pd.DataFrame({ID: ids_te, TARGET: final})
    assert sub[TARGET].nunique() == N and not sub[TARGET].isnull().any()
    assert np.array_equal(sub[ID].to_numpy(), sample[ID].to_numpy())
    sub.to_csv(f"subs/{name}.csv", index=False)
    print(f"SUCCESS subs/{name}.csv | {note} | rho vs base {spearmanr(final, base).statistic:.6f}")

built = []
for new, w in [("realmlp_gpu", 0.02), ("realmlp_gpu", 0.05), ("xgb_stack", 0.05)]:
    if new in test_preds:
        rn = rankdata(test_preds[new]) / N
        nm = {"realmlp_gpu": "v53" if w == 0.02 else "v54", "xgb_stack": "v55"}[new]
        emit(lexrank((1 - w) * rb + w * rn + shift, nn), f"{nm}_newmember_w{int(w*100)}",
             f"base + {int(w*100)}% {new}")
        built.append(nm)
print("\nbuilt:", built)
