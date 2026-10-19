import pandas as pd
import numpy as np
from scipy.stats import rankdata, spearmanr

print("--- BUILDING V29: TRI-ANCHOR APEX GRANDMASTER BLEND ---")

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

# Load all 5 diverse components
p_taeyang = pd.read_csv("scratch/kernels_pulled/taeyangg4_094650_out/submission.csv")[TARGET].to_numpy(dtype=float)
p_jazivxt = pd.read_csv("scratch/kernels_pulled/jazivxt_zoom_out/submission.csv")[TARGET].to_numpy(dtype=float)
p_stacker = np.load("scratch/test_meta_stacker_10way.npy")
p_v18 = pd.read_csv("subs/v18_nina_honest_residual_tiebreaker.csv")[TARGET].to_numpy(dtype=float)
p_deotte = pd.read_csv("scratch/deotte_gpu_out/sub_xgb_deotte_simpson.csv")[TARGET].to_numpy(dtype=float)
p_rmlp = pd.read_csv("scratch/realmlp_out/submission.csv")[TARGET].to_numpy(dtype=float)

r_taeyang = rk01(p_taeyang)
r_jazivxt = rk01(p_jazivxt)
r_stacker = rk01(p_stacker)
r_v18 = rk01(p_v18)
r_deotte = rk01(p_deotte)
r_rmlp = rk01(p_rmlp)

print("Pairwise Spearman rank correlations with Taeyang 0.94650:")
print(f"  Jazivxt:   {spearmanr(r_taeyang, r_jazivxt)[0]:.6f}")
print(f"  Stacker:   {spearmanr(r_taeyang, r_stacker)[0]:.6f}")
print(f"  v18 Nina:  {spearmanr(r_taeyang, r_v18)[0]:.6f}")
print(f"  Deotte:    {spearmanr(r_taeyang, r_deotte)[0]:.6f}")
print(f"  RealMLP:   {spearmanr(r_taeyang, r_rmlp)[0]:.6f}")

# Domain masks
income = pd.to_numeric(test["Annual_Income_USD"], errors="coerce").to_numpy()
commute = pd.to_numeric(test["Daily_Commute_km"], errors="coerce").to_numpy()
subsidy_no = (test["Subsidy_Available"].astype(str) == "No").to_numpy()
env_1 = (pd.to_numeric(test["Environmental_Concern_Level"], errors="coerce") == 1.0).to_numpy()
anx_med_high = test["Range_Anxiety_Level"].isin(["Medium", "High"]).to_numpy()

mask_upper = income >= 170537.0
mask_dead_inc = (income >= 31004.0) & (income <= 41970.0)
mask_commute = commute >= 83.0
mask_30k_zero = (income == 30000.0) & subsidy_no & (env_1 | anx_med_high)

print(f"\nMask summary: upper={mask_upper.sum()}, dead_inc={mask_dead_inc.sum()}, commute={mask_commute.sum()}, 30k_zero={mask_30k_zero.sum()}")

# Apex formulation:
# 60% Taeyang (0.94650) + 30% Jazivxt pass-2 (0.94649) + 4% 10-Model Stacker + 3% v18 Nina + 2% Deotte Simpson + 1% RealMLP
blend = (
    0.60 * r_taeyang
    + 0.30 * r_jazivxt
    + 0.04 * r_stacker
    + 0.03 * r_v18
    + 0.02 * r_deotte
    + 0.01 * r_rmlp
)

# Apply safe boundary calibrations:
blend[mask_upper] += 10.0
blend[mask_dead_inc] -= 10.0
blend[mask_commute] -= 5.0
blend[mask_30k_zero] -= 5.0

# Hierarchical lexicographical ranking via RealMLP
final_ranks = lexsort_ranking(blend, p_rmlp)

sub_v29 = pd.DataFrame({
    ID: test[ID].to_numpy(),
    TARGET: final_ranks,
})

# Verifications
assert len(sub_v29) == len(sample), "Row count mismatch!"
assert sub_v29[ID].equals(sample[ID]), "ID mismatch!"
assert not sub_v29[TARGET].isnull().any(), "Null values detected!"
assert sub_v29[TARGET].nunique() == len(sample), "Ties detected!"
assert sub_v29[TARGET].min() >= 0.0 and sub_v29[TARGET].max() <= 1.0, "Out of bounds!"

sub_v29.to_csv("subs/v29_tri_anchor_apex_grandmaster.csv", index=False)
print("\nSUCCESS: subs/v29_tri_anchor_apex_grandmaster.csv created successfully!")
print(f"Ranks min: {sub_v29[TARGET].min():.8f}, max: {sub_v29[TARGET].max():.8f}, unique: {sub_v29[TARGET].nunique():,}")
print(f"Spearman rank correlation with Taeyang 0.94650: {np.corrcoef(sub_v29[TARGET], r_taeyang)[0, 1]:.6f}")
print(f"Spearman rank correlation with Jazivxt pass-2:  {np.corrcoef(sub_v29[TARGET], r_jazivxt)[0, 1]:.6f}")
