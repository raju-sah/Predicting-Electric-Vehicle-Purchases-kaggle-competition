import pandas as pd
import numpy as np
from scipy.stats import rankdata

print("--- TESTING DUAL-ANCHOR GRANDMASTER BLEND ---")

TARGET = "Will_Buy_EV"
ID = "id"

def rk01(arr: np.ndarray) -> np.ndarray:
    return (rankdata(arr, method="ordinal") - 0.5) / len(arr)

def lexsort_ranking(primary: np.ndarray, secondary: np.ndarray) -> np.ndarray:
    order = np.lexsort((secondary, primary))
    ranks = np.empty(len(order), dtype=np.int64)
    ranks[order] = np.arange(1, len(order) + 1)
    return (ranks - 0.5) / len(ranks)

test = pd.read_csv("test.csv")
sample = pd.read_csv("sample_submission.csv")

p_v26 = pd.read_csv("subs/v26_taeyang_lexsort_master.csv")[TARGET].to_numpy(dtype=float)
p_v18 = pd.read_csv("subs/v18_nina_honest_residual_tiebreaker.csv")[TARGET].to_numpy(dtype=float)
p_stacker = np.load("scratch/test_meta_stacker_10way.npy")
p_rmlp = pd.read_csv("scratch/realmlp_out/submission.csv")[TARGET].to_numpy(dtype=float)
p_deotte = pd.read_csv("scratch/deotte_gpu_out/sub_xgb_deotte_simpson.csv")[TARGET].to_numpy(dtype=float)

r_v26 = rk01(p_v26)
r_v18 = rk01(p_v18)
r_stacker = rk01(p_stacker)
r_rmlp = rk01(p_rmlp)
r_deotte = rk01(p_deotte)

income = pd.to_numeric(test["Annual_Income_USD"], errors="coerce").to_numpy()
commute = pd.to_numeric(test["Daily_Commute_km"], errors="coerce").to_numpy()
subsidy_no = (test["Subsidy_Available"].astype(str) == "No").to_numpy()
env_1 = (pd.to_numeric(test["Environmental_Concern_Level"], errors="coerce") == 1.0).to_numpy()
anx_med_high = test["Range_Anxiety_Level"].isin(["Medium", "High"]).to_numpy()

mask_upper = income >= 170537.0
mask_dead_inc = (income >= 31004.0) & (income <= 41970.0)
mask_commute = commute >= 83.0
mask_30k_zero = (income == 30000.0) & subsidy_no & (env_1 | anx_med_high)

print(f"Mask counts: Upper={mask_upper.sum()}, DeadInc={mask_dead_inc.sum()}, Commute={mask_commute.sum()}, 30kZero={mask_30k_zero.sum()}")

# Blend: 85% v26 + 10% v18 + 3% Stacker + 2% Deotte
blend = 0.85 * r_v26 + 0.10 * r_v18 + 0.03 * r_stacker + 0.02 * r_deotte

# Apply safe boundary calibrations:
blend[mask_upper] += 10.0
blend[mask_dead_inc] -= 10.0
blend[mask_commute] -= 5.0
blend[mask_30k_zero] -= 5.0

# Hierarchical lexicographical ranking via RealMLP
final_ranks = lexsort_ranking(blend, p_rmlp)

sub_v28 = pd.DataFrame({
    ID: test[ID].to_numpy(),
    TARGET: final_ranks,
})

print(f"Verification: rows={len(sub_v28)}, unique={sub_v28[TARGET].nunique():,}, min={sub_v28[TARGET].min():.8f}, max={sub_v28[TARGET].max():.8f}")
print(f"Correlation with v26: {np.corrcoef(final_ranks, r_v26)[0, 1]:.6f}")
print(f"Correlation with v18: {np.corrcoef(final_ranks, r_v18)[0, 1]:.6f}")

sub_v28.to_csv("subs/v28_dual_anchor_lexsort_blend.csv", index=False)
print("Saved to subs/v28_dual_anchor_lexsort_blend.csv")
