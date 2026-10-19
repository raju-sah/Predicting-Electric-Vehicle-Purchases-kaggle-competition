# v38: pure anchor + 12% hill-climbed stack (v37b weights, includes the new najiama
#      Sep-14 members + megayak six views + RealMLP-G), boundary rules, lexsort.
# v39: same in probit space with the injection passed through a 0.74 power residual
#      (merges the historically best v33 probit + v34 power-residual perturbations
#      with the new stronger stack).
import numpy as np
import pandas as pd
from scipy.stats import rankdata, norm, spearmanr

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

# pure 0.94649 anchor
sub_sep14 = load("scratch/kernels_pulled/jazivxt_zoom_sep14/submission.csv")
pred_lgb = load("scratch/kernels_pulled/jazivxt_zoom_sep14/submission_lightgbm.csv")
anchor = (sub_sep14 - 0.20 * pred_lgb) / 0.80

# RealMLP-G 3-seed (continuous secondary + injection component)
nn = load("scratch/pull_sep16/ds_megayak6/test_realmlp_g.csv", col="G_realmlp_3seed")

# hill-climbed stack test predictions (v37b weights applied in rank space)
stack_test = np.load("test_v37b_stack.npy")
assert len(stack_test) == N

# boundary rules
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

# ---------------- v38: 88% anchor / 12% hill-climbed stack ----------------
blend38 = 0.88 * lexrank(anchor, nn) + 0.12 * rk(stack_test)
final38 = lexrank(blend38 + shift, nn)
sub38 = pd.DataFrame({ID: ids, TARGET: final38})
sample = pd.read_csv("sample_submission.csv")
assert np.array_equal(sub38[ID].to_numpy(), sample[ID].to_numpy())
assert sub38[TARGET].nunique() == N and not sub38[TARGET].isnull().any()
sub38.to_csv("subs/v38_apex_hillclimb12.csv", index=False)
print("SUCCESS subs/v38_apex_hillclimb12.csv",
      f"| rho vs anchor {spearmanr(final38, anchor).statistic:.6f}")

# ---------------- v39: probit-space apex with 0.74 power injection ----------------
z_anchor = norm.ppf(np.clip(lexrank(anchor, nn), 1e-6, 1 - 1e-6))
inj = 0.60 * rk(stack_test) + 0.40 * rk(nn)
inj_power = np.sign(inj - 0.5) * np.abs(inj - 0.5) ** 0.74 + 0.5   # v34-style power residual
z_inj = norm.ppf(np.clip(inj_power, 1e-6, 1 - 1e-6))
z_blend = 0.88 * z_anchor + 0.12 * z_inj
p_final = norm.cdf(z_blend) + shift * 1e-6
final39 = lexrank(p_final, nn)
sub39 = pd.DataFrame({ID: ids, TARGET: final39})
assert np.array_equal(sub39[ID].to_numpy(), sample[ID].to_numpy())
assert sub39[TARGET].nunique() == N and not sub39[TARGET].isnull().any()
sub39.to_csv("subs/v39_apex_probit_power12.csv", index=False)
print("SUCCESS subs/v39_apex_probit_power12.csv",
      f"| rho vs anchor {spearmanr(final39, anchor).statistic:.6f}",
      f"| rho vs v38 {spearmanr(final39, final38).statistic:.6f}")
