# v36: Anchor-preserving apex on the pure 0.94649 shared anchor, following the
# taeyangg4 lexsort recipe and megayak's forkable rebuild, with upgraded slots:
# stack = 0.50*six-view trees + 0.25*RealMLP-G(3seed) + 0.25*najiama XGB 10-fold
# injection total = 10% (6% stack + 4% nn), inside megayak's proven safe envelope.
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

TARGET, ID = "Will_Buy_EV", "id"
test = pd.read_csv("test.csv")
ids = test[ID].to_numpy()
N = len(test)

def rk(v):
    return rankdata(v, method="average") / len(v)

def lexsort_ranking(primary, secondary):
    o = np.lexsort((secondary, primary))
    r = np.empty(len(o), dtype=np.int64)
    r[o] = np.arange(1, len(o) + 1)
    return (r - 0.5) / len(r)

def load(path, col=None):
    d = pd.read_csv(path).sort_values(ID)
    assert np.array_equal(d[ID].to_numpy(), ids), path
    c = col if col else [c for c in d.columns if c != ID][0]
    return d[c].to_numpy(dtype=float)

# 1) pure 0.94649 anchor (reconstructed: shared = (sub - 0.20*lgb) / 0.80)
sub_sep14 = load("scratch/kernels_pulled/jazivxt_zoom_sep14/submission.csv")
pred_lgb = load("scratch/kernels_pulled/jazivxt_zoom_sep14/submission_lightgbm.csv")
shared_094649 = (sub_sep14 - 0.20 * pred_lgb) / 0.80

# 2) stack components (all ranked)
sv = pd.read_csv("scratch/pull_sep16/ds_megayak6/test_six_views.csv").sort_values(ID)
views = {"A_lgbm_triple_te_digits_3seed": .2, "B_xgb_on_A_features": .2,
         "C_no_digits_windows_lift_sm2_30_300": .1, "D_no_exact_key_ladder_windows": .2,
         "E_ladder25_250_2500_lift_sm5_50_500": .1, "F_exact_rate_as_init_score": .2}
trees = rk(sum(w * rk(sv[c].to_numpy(dtype=float)) for c, w in views.items()))
nn = load("scratch/pull_sep16/ds_megayak6/test_realmlp_g.csv", col="G_realmlp_3seed")
xgb = load("scratch/pull_sep16/ds_najiama/XGBoost_Triple_TE_10folds_test.csv")
stack = rk(0.50 * trees + 0.25 * rk(nn) + 0.25 * rk(xgb))

# 3) recipe: 90% lexrank(anchor, nn) + 6% stack + 4% nn, boundary shifts, final lexsort
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

blend = 0.90 * lexsort_ranking(shared_094649, nn) + 0.06 * stack + 0.04 * rk(nn)
final = lexsort_ranking(blend + shift, nn)

sub = pd.DataFrame({ID: ids, TARGET: final})
assert len(sub) == N and not sub[TARGET].isnull().any()
assert sub[TARGET].nunique() == N, "ties detected"
sample = pd.read_csv("sample_submission.csv")
assert np.array_equal(sub[ID].to_numpy(), sample[ID].to_numpy()), "id order mismatch"
sub.to_csv("subs/v36_apex_upgraded_slots.csv", index=False)
print("SUCCESS subs/v36_apex_upgraded_slots.csv")
print(f"spearman vs anchor: {spearmanr(final, shared_094649).statistic:.6f}")
print(f"spearman vs v33:    {spearmanr(final, load('subs/v33_probit_latent_apex.csv')).statistic:.6f}")
