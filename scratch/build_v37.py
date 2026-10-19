# v37: megayak's published OOF-library stack, rebuilt exactly from his public files:
# 0.57*six-view ensemble + 0.29*RealMLP-G(3seed) + 0.14*najiama XGB-10fold,
# percentile-rank space, plus the four boundary rules, final lexsort tie-break.
# Public LB of this recipe: 0.94644 — our independent second anchor.
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

sv = pd.read_csv("scratch/pull_sep16/ds_megayak6/test_six_views.csv").sort_values(ID)
six = rk(sv["ensemble"].to_numpy(dtype=float))
nn = load("scratch/pull_sep16/ds_megayak6/test_realmlp_g.csv", col="G_realmlp_3seed")
xgb = load("scratch/pull_sep16/ds_najiama/XGBoost_Triple_TE_10folds_test.csv")

blend = rk(0.57 * six + 0.29 * rk(nn) + 0.14 * rk(xgb))

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

final = lexsort_ranking(blend + shift, nn)

sub = pd.DataFrame({ID: ids, TARGET: final})
sample = pd.read_csv("sample_submission.csv")
assert np.array_equal(sub[ID].to_numpy(), sample[ID].to_numpy())
assert sub[TARGET].nunique() == N and not sub[TARGET].isnull().any()
sub.to_csv("subs/v37_megayak_oof_stack.csv", index=False)
print("SUCCESS subs/v37_megayak_oof_stack.csv")

# reference correlations
anchor = (load("scratch/kernels_pulled/jazivxt_zoom_sep14/submission.csv")
          - 0.20 * load("scratch/kernels_pulled/jazivxt_zoom_sep14/submission_lightgbm.csv")) / 0.80
print(f"spearman vs pure anchor: {spearmanr(final, anchor).statistic:.6f}")
print(f"spearman vs v36:         {spearmanr(final, load('subs/v36_apex_upgraded_slots.csv')).statistic:.6f}")
