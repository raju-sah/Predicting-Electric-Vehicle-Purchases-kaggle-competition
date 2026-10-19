# v40-v42: apex curve on jazivxt's exact 0.94651 base (submission_latest_best.csv,
# SHA-verified against ranked_catalog_latest.csv, rho=0.99998 vs our old anchor).
# Form = proven v38 mechanics: (1-w)*lexrank(base, nn) + w*rk(hill-climbed stack),
# + boundary shifts, final lexsort tie-break with nn.
# Curve: v40 w=0.10, v41 w=0.12 (pure A/B vs v38 old-base), v42 w=0.15 (envelope push).
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

TARGET, ID = "Will_Buy_EV", "id"
test = pd.read_csv("test.csv")
ids = test[ID].to_numpy()
N = len(test)

def rk(v):
    return rankdata(v, method="average") / len(v)

def lexrank(primary, secondary):
    o = np.lexsort((secondary, primary))
    r = np.empty(len(o), dtype=np.int64)
    r[o] = np.arange(1, len(o) + 1)
    return (r - 0.5) / len(r)

def load(path, col=None):
    d = pd.read_csv(path).sort_values(ID)
    assert np.array_equal(d[ID].to_numpy(), ids), path
    c = col if col else [c for c in d.columns if c != ID][0]
    return d[c].to_numpy(dtype=float)

# 1) NEW base: jazivxt's exact 0.94651 scored file (sha-verified earlier)
base = load("scratch/pull_sep16/jz_meta/submission_latest_best.csv")

# RealMLP-G 3-seed (continuous secondary + injection component)
nn = load("scratch/pull_sep16/ds_megayak6/test_realmlp_g.csv", col="G_realmlp_3seed")

# hill-climbed stack test predictions (v37b weights, Sep-14 members included)
stack_test = np.load("test_v37b_stack.npy")
assert len(stack_test) == N

# boundary rules (proven across v32-v38)
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
print("rows shifted:", int((shift != 0).sum()))

sample = pd.read_csv("sample_submission.csv")
base_rk = rk(base)

for w, name in [(0.10, "v40"), (0.12, "v41"), (0.15, "v42")]:
    blend = (1 - w) * lexrank(base, nn) + w * rk(stack_test)
    final = lexrank(blend + shift, nn)
    sub = pd.DataFrame({ID: ids, TARGET: final})
    assert len(sub) == N and not sub[TARGET].isnull().any()
    assert sub[TARGET].nunique() == N, "ties detected"
    assert np.array_equal(sub[ID].to_numpy(), sample[ID].to_numpy()), "id order mismatch"
    out = f"subs/{name}_apex_jz_base_w{int(w*100):02d}.csv"
    sub.to_csv(out, index=False)
    print(f"SUCCESS {out} | w={w} | rho vs base {spearmanr(final, base).statistic:.6f}")

# cross-checks
v41 = load("subs/v41_apex_jz_base_w12.csv")
v38 = load("subs/v38_apex_hillclimb12.csv")
print(f"v41 vs v38 (A/B pair): {spearmanr(v41[TARGET].to_numpy(), v38[TARGET].to_numpy()).statistic:.6f}")
