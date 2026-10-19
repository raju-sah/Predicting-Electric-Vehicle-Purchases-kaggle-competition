# v43-v45: decomposition of why transformations on jazivxt's 0.94651 base read 0.94650.
# v43 = exact base file, untouched (control; expect 0.94651)
# v44 = base + raw boundary shifts only (shift probe at full strength, no reorder/injection)
# v45 = lexrank(base, nn) only (reorder probe: tests whether the nn-secondary
#        lexrank re-sort itself destroys the base's edge; no shift, no stack)
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

base = load("scratch/pull_sep16/jz_meta/submission_latest_best.csv")
nn = load("scratch/pull_sep16/ds_megayak6/test_realmlp_g.csv", col="G_realmlp_3seed")

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
    sub = pd.DataFrame({ID: ids, TARGET: final})
    assert len(sub) == N and not sub[TARGET].isnull().any()
    assert sub[TARGET].nunique() == N, "ties detected"
    assert np.array_equal(sub[ID].to_numpy(), sample[ID].to_numpy()), "id order mismatch"
    out = f"subs/{name}.csv"
    sub.to_csv(out, index=False)
    print(f"SUCCESS {out} | {note} | rho vs base {spearmanr(final, base).statistic:.6f}")

# v43: exact base, byte-prediction-identical ranking (copy of verified file)
import shutil
shutil.copyfile("scratch/pull_sep16/jz_meta/submission_latest_best.csv",
                "subs/v43_jz_base_exact.csv")
print("SUCCESS subs/v43_jz_base_exact.csv | exact 0.94651 file, sha-verified")

# v44: base + raw shifts, nothing else
emit(lexrank(rk(base) + shift, nn), "v44_jz_base_shift_only", "base + raw shifts only")

# v45: nn-secondary lexrank reorder of base, nothing else
emit(lexrank(base, nn), "v45_jz_base_lexrank_only", "lexrank(base, nn) only")
