# v37b: hill-climbed stack over ALL honest OOF members (seed-42 fold-mean criterion).
# Members: megayak six views + ensemble, RealMLP-G(3seed), najiama Sep-14 pack
# (XGB-10f, XGB-5f, PureLGBM V3/V1, Sergey, 01_blend), my legacy OOFs, golem h/i,
# and v35 if present. Greedy forward selection with coordinate refinement.
import numpy as np
import pandas as pd
import glob, os
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

train = pd.read_csv("train.csv")
y = (train["Will_Buy_EV"] == "Yes").astype(np.int8).to_numpy()
N = len(y)
from sklearn.model_selection import StratifiedKFold
folds = np.zeros(N, dtype=int)
for k, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(N), y)):
    folds[va] = k

def rk(v):
    return rankdata(v, method="average") / len(v)

def load_col(path, col=None, id_col="id"):
    d = pd.read_csv(path)
    c = col if col else [x for x in d.columns if x != id_col][0]
    return d.sort_values(id_col)[c].to_numpy(dtype=float)

OOF, TEST = {}, {}

sv_oof = pd.read_csv("scratch/pull_sep16/ds_megayak6/oof_six_views.csv")
sv_test = pd.read_csv("scratch/pull_sep16/ds_megayak6/test_six_views.csv")
for c in sv_oof.columns:
    if c in ("id", "fold", "Will_Buy_EV"):
        continue
    OOF["m6_" + c] = sv_oof[c].to_numpy(dtype=float)
    TEST["m6_" + c] = sv_test[c].to_numpy(dtype=float)

o = load_col("scratch/pull_sep16/ds_megayak6/oof_realmlp_g.csv", col="G_realmlp_3seed")
t = load_col("scratch/pull_sep16/ds_megayak6/test_realmlp_g.csv", col="G_realmlp_3seed")
OOF["m6_realmlp3"] = o; TEST["m6_realmlp3"] = t

for fn, nm in [("XGBoost_Triple_TE_10folds", "naji_xgb10"), ("XGBoost_Triple_TE_5folds", "naji_xgb5"),
               ("Pure LGBM_V3", "naji_v3"), ("Pure LGBM_V1", "naji_v1"),
               ("Sergey_LGBM", "naji_sergey"), ("01_blend", "naji_blend")]:
    OOF[nm] = load_col(f"scratch/pull_sep16/ds_najiama/{fn}_oof.csv")
    tpath = f"scratch/pull_sep16/ds_najiama/{fn}_test.csv"
    if not os.path.exists(tpath):
        tpath = f"scratch/pull_sep16/ds_najiama/{fn.replace('01_blend', '01')}_submission.csv"
    TEST[nm] = load_col(tpath)

for p in sorted(glob.glob("oof_*.npy")):
    name = os.path.basename(p)[4:-4]
    if name.startswith("v35"):
        continue
    OOF["my_" + name] = np.load(p)
    tp = "test_" + name + ".npy"
    if os.path.exists(tp):
        TEST["my_" + name] = np.load(tp)

for nm in ["G_h", "G_i"]:
    OOF["my_" + nm.lower()] = np.load(f"scratch/golem_lib/oof_{nm[2:].lower()}.npy")
    TEST["my_" + nm.lower()] = np.load(f"scratch/golem_lib/test_{nm[2:].lower()}.npy")

if os.path.exists("oof_v35_zoom_max.npy"):
    OOF["v35"] = np.load("oof_v35_zoom_max.npy")
    TEST["v35"] = np.load("test_v35_zoom_max.npy")
    OOF["v35_p1"] = np.load("oof_v35_zoom_max_pass1.npy")
    TEST["v35_p1"] = np.load("test_v35_zoom_max_pass1.npy")

memory_oof = pd.read_csv("scratch/pull_sep18/memory_recipe_out/oof_predictions.csv").sort_values("id")
memory_test = pd.read_csv("scratch/pull_sep18/memory_recipe_out/submission_model_only.csv").sort_values("id")
assert np.array_equal(memory_oof["id"].to_numpy(), train["id"].to_numpy())
OOF["memory_residual"] = memory_oof["oof_final"].to_numpy(dtype=float)
TEST["memory_residual"] = memory_test["Will_Buy_EV"].to_numpy(dtype=float)

# keep complete members with pooled AUC >= 0.944
pool = {}
for k in OOF:
    if k not in TEST:
        continue
    v, t = OOF[k], TEST[k]
    if v is None or t is None or len(v) != N or np.isnan(v).any() or np.isnan(t).any():
        continue
    if len(t) != 286571:
        continue
    a = roc_auc_score(y, v)
    if a >= 0.9440:
        pool[k] = (rk(v), rk(t), a)

print("=== pool ===")
for k, (_, _, a) in sorted(pool.items(), key=lambda kv: -kv[1][2]):
    print(f"{k:32s} {a:.6f}")

keys = list(pool.keys())
R = np.column_stack([pool[k][0] for k in keys])
RT = np.column_stack([pool[k][1] for k in keys])
pf = np.array([[roc_auc_score(y[folds == f], R[:, i][folds == f]) for f in range(5)] for i in range(len(keys))])

def fold_mean(w):
    blend = R @ w
    return np.mean([roc_auc_score(y[folds == f], blend[folds == f]) for f in range(5)])

# greedy forward (unnormalized weights via simple accumulation in rank space)
w = np.zeros(len(keys)); cur = np.zeros(N)
for step in range(10):
    best_j, best_m = -1, -1
    for j in range(len(keys)):
        w2 = w.copy(); w2[j] += 1.0
        m = fold_mean(w2 / w2.sum())
        if m > best_m:
            best_m, best_j = m, j
    w[best_j] += 1.0
    wnorm = w / w.sum()
    print(f"step {step+1}: + {keys[best_j]:30s} fold-mean {best_m:.6f}  w={np.round(wnorm,3).tolist()}")

# coordinate refinement (keep simplex, 3 passes)
wnorm = w / w.sum()
for it in range(3):
    improved = False
    for j in range(len(keys)):
        for delta in (0.02, 0.05, 0.10):
            w2 = wnorm.copy()
            take = min(delta, w2[j])
            donors = [i for i in range(len(keys)) if i != j and w2[i] > 0.01]
            if not donors:
                continue
            d = np.random.RandomState(0).choice(donors)
            w2[j] += take; w2[d] -= take
            m2 = fold_mean(w2)
            if m2 > fold_mean(wnorm) + 1e-7:
                wnorm = w2; improved = True
    print(f"refine pass {it+1}: fold-mean {fold_mean(wnorm):.6f}")
    if not improved:
        break

print("final weights:")
for k, ww in sorted(zip(keys, wnorm), key=lambda kv: -kv[1]):
    if ww > 0.004:
        print(f"  {k:30s} {ww:.3f}")

final_oof = R @ wnorm
full = roc_auc_score(y, final_oof)
pfm = fold_mean(wnorm)
print(f"STACK pooled {full:.6f} fold-mean {pfm:.6f}")

test_stack = RT @ wnorm
np.save("oof_v56_memory_stack.npy", final_oof)
np.save("test_v56_memory_stack.npy", test_stack)
print("saved oof/test_v56_memory_stack.npy")
