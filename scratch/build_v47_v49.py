# v47-v49 candidates: ensemble the 12 distinct local optima mined from jazivxt's
# scored archive (G0 = 0.94651 trio rank-identical; G1..G11 = distinct 0.94650 files).
# Hypothesis: bagging around the peak — mean/median of neighboring optima can beat
# the single best. All in rank space, final lexrank(., nn) tie-break (proven free).
import numpy as np, pandas as pd, json
from scipy.stats import rankdata, spearmanr

TARGET, ID = "Will_Buy_EV", "id"
sample = pd.read_csv("sample_submission.csv")
ids = sample[ID].to_numpy()
N = len(ids)

def lexrank(primary, secondary):
    o = np.lexsort((secondary, primary))
    r = np.empty(len(o), dtype=np.int64)
    r[o] = np.arange(1, len(o) + 1)
    return (r - 0.5) / len(r)

meta = json.load(open("scratch/pull_sep17/tier_groups.json"))
R = []
for m in meta:
    d = pd.read_csv(m["path"]).sort_values(ID)
    assert np.array_equal(d[ID].to_numpy(), ids)
    R.append(rankdata(d[TARGET].to_numpy(dtype=float)))
R = np.array(R)  # 12 x N
print("rank matrix:", R.shape)

base = pd.read_csv("scratch/pull_sep16/jz_meta/submission_latest_best.csv").sort_values(ID)
base = base[TARGET].to_numpy(dtype=float)
rb = rankdata(base)
nn = pd.read_csv("scratch/pull_sep16/ds_megayak6/test_realmlp_g.csv").sort_values(ID)
nn = nn["G_realmlp_3seed"].to_numpy(dtype=float)

def emit(final, name, note):
    sub = pd.DataFrame({ID: ids, TARGET: final})
    assert sub[TARGET].nunique() == N and not sub[TARGET].isnull().any()
    assert np.array_equal(sub[ID].to_numpy(), sample[ID].to_numpy())
    out = f"subs/{name}.csv"
    sub.to_csv(out, index=False)
    print(f"SUCCESS {out} | {note} | rho vs base {spearmanr(final, base).statistic:.6f}")

# v47: plain mean of all 12 optima (G0 counted once)
m47 = np.mean(R, axis=0)
emit(lexrank(m47, nn), "v47_optima_family_mean12", "mean of 12 distinct optima")

# v48: median of all 12 (robust to one drifting optimum)
m48 = np.median(R, axis=0)
emit(lexrank(m48, nn), "v48_optima_family_median12", "median of 12 distinct optima")

# v49: base-pull mean — 50% G0 + 50% mean(G1..G11), anchored toward the known peak
mean_rest = np.mean(R[1:], axis=0)
m49 = 0.5 * R[0] + 0.5 * mean_rest
emit(lexrank(m49, nn), "v49_base_pull_family", "0.5*G0(0.94651) + 0.5*mean(G1..G11)")

# diagnostics: how far do the blends move from base?
for nm, m in [("v47", m47), ("v48", m48), ("v49", m49)]:
    d = np.abs(m - rb)
    print(f"{nm}: mean|drift|={d.mean():.1f} ranks, max={d.max():.0f}, rows>50 ranks: {(d>50).sum()}")
