# v46: last probe of the day on the secured 0.94651 base family.
# v44 showed 1x boundary shifts are exactly free (0.94651). This probes 2x strength:
# if shifts encode true generator structure the LB may tip +0.00001; if they are
# mildly harmful, 1x would already have dropped. Either way we learn the sign.
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

final = lexrank(rk(base) + 2.0 * shift, nn)
sub = pd.DataFrame({ID: ids, TARGET: final})
assert sub[TARGET].nunique() == N and not sub[TARGET].isnull().any()
sample = pd.read_csv("sample_submission.csv")
assert np.array_equal(sub[ID].to_numpy(), sample[ID].to_numpy())
sub.to_csv("subs/v46_jz_base_shift2x.csv", index=False)
print("SUCCESS subs/v46_jz_base_shift2x.csv | rho vs base",
      f"{spearmanr(final, base).statistic:.6f}")
