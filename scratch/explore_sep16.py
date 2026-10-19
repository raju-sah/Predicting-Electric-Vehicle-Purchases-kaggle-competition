import numpy as np, pandas as pd, glob, os
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

ROOT = "."
train = pd.read_csv("train.csv")
y = (train["Will_Buy_EV"] == "Yes").astype(np.int8).to_numpy()
N = len(y)

# Frozen folds (seed-42 5-fold stratified, identical construction everywhere)
from sklearn.model_selection import StratifiedKFold
folds = np.zeros(N, dtype=int)
skf = StratifiedKFold(5, shuffle=True, random_state=42)
for k, (_, va) in enumerate(skf.split(np.zeros(N), y)):
    folds[va] = k

def rk(v):
    return rankdata(v, method="average") / len(v)

def load_csv_col(path, col=None, id_col="id"):
    d = pd.read_csv(path)
    target_cols = [c for c in d.columns if c != id_col]
    c = col if col else target_cols[0]
    if c not in d.columns:
        return None
    d = d.sort_values(id_col)
    return d[c].to_numpy(dtype=float)

OOF, TEST = {}, {}

# --- golem library (frozen-fold, honest) ---
for p in sorted(glob.glob("scratch/golem_lib/oof_*.npy")):
    name = os.path.basename(p)[4:-4]
    OOF["G_" + name] = np.load(p)
for p in sorted(glob.glob("scratch/golem_lib/test_*.npy")):
    name = os.path.basename(p)[5:-4]
    TEST["G_" + name] = np.load(p)

# --- najiama Sep-14 pack ---
for fn in os.listdir("scratch/pull_sep16/ds_najiama"):
    name = fn.replace(".csv", "")
    if name.endswith("_oof"):
        OOF["N_" + name[:-4]] = load_csv_col(f"scratch/pull_sep16/ds_najiama/{fn}")
    elif "test" in fn or "submission" in fn:
        TEST["N_" + name.replace("test", "").replace("submission", "")] = load_csv_col(f"scratch/pull_sep16/ds_najiama/{fn}")

# --- megayak six views + realmlp_g (test has id col, oof too) ---
sv_oof = pd.read_csv("scratch/pull_sep16/ds_megayak6/oof_six_views.csv")
sv_test = pd.read_csv("scratch/pull_sep16/ds_megayak6/test_six_views.csv")
for c in sv_oof.columns:
    if c in ("id", "fold"):
        continue
    OOF["M6_" + c] = sv_oof[c].to_numpy(dtype=float)
    if c in sv_test.columns:
        TEST["M6_" + c] = sv_test[c].to_numpy(dtype=float)
OOF["M6_realmlp_g"] = load_csv_col("scratch/pull_sep16/ds_megayak6/oof_realmlp_g.csv")
TEST["M6_realmlp_g"] = load_csv_col("scratch/pull_sep16/ds_megayak6/test_realmlp_g.csv")

# --- my legacy OOFs ---
for p in sorted(glob.glob("oof_*.npy")):
    name = os.path.basename(p)[4:-4]
    OOF["MY_" + name] = np.load(p)

# --- reconstructed anchor 0.94649 (test-only) ---
anchor_test = load_csv_col("scratch/reconstructed_shared_094649.csv")
r_anchor = rk(anchor_test)

# --- top submission files for architecture comparison ---
subs = {}
for f in ["subs/v26_taeyang_lexsort_master.csv", "subs/v33_probit_latent_apex.csv",
          "subs/v11_megayak_4view_daily.csv", "subs/v18_nina_honest_residual_tiebreaker.csv"]:
    subs[os.path.basename(f)] = load_csv_col(f)

print(f"y positive rate {y.mean():.4f}, N={N}")
print("\n=== OOF members: overall AUC + per-fold AUC ===")
rows = []
for k, v in OOF.items():
    if v is None or len(v) != N or np.isnan(v).any():
        rows.append((k, np.nan, [np.nan] * 5))
        continue
    a = roc_auc_score(y, v)
    pf = [roc_auc_score(y[folds == k_], v[folds == k_]) for k_ in range(5)]
    rows.append((k, a, pf))
rows.sort(key=lambda r: -(r[1] if not np.isnan(r[1]) else 0))
for k, a, pf in rows:
    print(f"{k:34s} OOF {a:.6f}  folds " + " ".join(f"{x:.5f}" for x in pf))

# Correlation matrix of the top members vs each other
top_names = [k for k, a, _ in rows if a >= 0.9400][:24]
print("\n=== Spearman correlations among top members (train OOF) ===")
keys = [k for k in top_names if k in OOF and OOF[k] is not None and not np.isnan(OOF[k]).any()]
M = np.zeros((len(keys), len(keys)))
for i in range(len(keys)):
    for j in range(len(keys)):
        M[i, j] = spearmanr(OOF[keys[i]], OOF[keys[j]]).statistic
df = pd.DataFrame(M, index=keys, columns=keys)
pd.set_option("display.width", 250, "display.max_columns", 50)
print(df.round(3).to_string())

# Greedy forward selection on per-fold AUC (5 obs) with replacement protection
print("\n=== Greedy forward selection (criterion: mean per-fold AUC) ===")
avail = {k: OOF[k] for k in keys}
sel, cur = [], None
for step in range(12):
    best_k, best_auc = None, -1
    for k in avail:
        if k in sel:
            continue
        cand = rk(avail[k]) if cur is None else cur + rk(avail[k])
        pf = [roc_auc_score(y[folds == f_], cand[folds == f_]) for f_ in range(5)]
        m = np.mean(pf)
        if m > best_auc:
            best_auc, best_k = m, k
    sel.append(best_k)
    cur = (rk(avail[best_k]) if len(sel) == 1 else cur + rk(avail[best_k]))
    full = roc_auc_score(y, cur)
    pf = [roc_auc_score(y[folds == f_], cur[folds == f_]) for f_ in range(5)]
    print(f"step {step+1}: + {best_k:32s} fold-mean {np.mean(pf):.6f} full {full:.6f} folds " +
          " ".join(f"{x:.5f}" for x in pf))

print("\n=== Test-side Spearman vs anchor (0.94649 shared) ===")
for k in keys:
    if k in TEST and TEST[k] is not None and len(TEST[k]) == len(anchor_test):
        r = spearmanr(TEST[k], anchor_test).statistic
        print(f"{k:34s} rho_test {r:.5f}")
for name, v in subs.items():
    print(f"SUB {name:50s} rho {spearmanr(v, anchor_test).statistic:.5f}")
