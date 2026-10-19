import pandas as pd
import numpy as np
from scipy.stats import rankdata

print("--- BUILDING V34: APEX POWER-RESIDUAL TIEBREAKER ENSEMBLE ---")

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

# 1. Load components
sub_sep14 = pd.read_csv("scratch/kernels_pulled/jazivxt_zoom_sep14/submission.csv")[TARGET].to_numpy(dtype=float)
pred_lgb = pd.read_csv("scratch/kernels_pulled/jazivxt_zoom_sep14/submission_lightgbm.csv")[TARGET].to_numpy(dtype=float)

# Pure unadulterated 0.94649 reference
shared_094649 = (sub_sep14 - 0.20 * pred_lgb) / 0.80

p_taeyang = pd.read_csv("scratch/kernels_pulled/taeyangg4_094650_out/submission.csv")[TARGET].to_numpy(dtype=float)
p_deotte = pd.read_csv("scratch/deotte_gpu_out/sub_xgb_deotte_simpson.csv")[TARGET].to_numpy(dtype=float)
p_stacker = np.load("scratch/test_meta_stacker_10way.npy")
p_rmlp = pd.read_csv("scratch/realmlp_out/submission.csv")[TARGET].to_numpy(dtype=float)
p_v18 = pd.read_csv("subs/v18_nina_honest_residual_tiebreaker.csv")[TARGET].to_numpy(dtype=float)

r_shared = rk01(shared_094649)
r_taeyang = rk01(p_taeyang)
r_deotte = rk01(p_deotte)
r_stacker = rk01(p_stacker)
r_rmlp = rk01(p_rmlp)
r_v18 = rk01(p_v18)

# 2. Master Macro Backbone: 50% pure shared + 50% Taeyang
r_master = 0.50 * r_shared + 0.50 * r_taeyang

# 3. Secondary Orthogonal Diversity Signal:
# 50% 10-Model Stacker + 30% Deotte Simpson + 20% RealMLP
r_secondary = 0.50 * r_stacker + 0.30 * r_deotte + 0.20 * r_rmlp

# 4. Nina Power-Residual Tiebreaker with gamma = 0.74 and alpha = 0.08:
delta = r_secondary - r_master
residual_term = np.sign(delta) * (np.abs(delta) ** 0.74)

# Blend: 92% master + 8% non-linear residual
blend = r_master + 0.08 * residual_term

# 5. Domain boundary masks
income = pd.to_numeric(test["Annual_Income_USD"], errors="coerce").to_numpy()
commute = pd.to_numeric(test["Daily_Commute_km"], errors="coerce").to_numpy()
subsidy_no = (test["Subsidy_Available"].astype(str) == "No").to_numpy()
env_1 = (pd.to_numeric(test["Environmental_Concern_Level"], errors="coerce") == 1.0).to_numpy()
anx_med_high = test["Range_Anxiety_Level"].isin(["Medium", "High"]).to_numpy()

mask_upper = income >= 170537.0
mask_dead_inc = (income >= 31004.0) & (income <= 41970.0)
mask_commute = commute >= 83.0
mask_30k_zero = (income == 30000.0) & subsidy_no & (env_1 | anx_med_high)

# Apply Boundary Calibrations:
blend[mask_upper] += 10.0
blend[mask_dead_inc] -= 10.0
blend[mask_commute] -= 5.0
blend[mask_30k_zero] -= 5.0

# Hierarchical Lexicographical Sorting via continuous RealMLP
final_ranks = lexsort_ranking(blend, p_rmlp)

sub_v34 = pd.DataFrame({
    ID: test[ID].to_numpy(),
    TARGET: final_ranks,
})

# Strict Verifications
assert len(sub_v34) == len(sample), "Row count mismatch!"
assert sub_v34[ID].equals(sample[ID]), "ID mismatch!"
assert not sub_v34[TARGET].isnull().any(), "Null values detected!"
assert sub_v34[TARGET].nunique() == len(sample), "Ties detected!"
assert sub_v34[TARGET].min() >= 0.0 and sub_v34[TARGET].max() <= 1.0, "Out of bounds!"

sub_v34.to_csv("subs/v34_apex_power_residual_ensemble.csv", index=False)
print("SUCCESS: subs/v34_apex_power_residual_ensemble.csv created successfully!")
print(f"Ranks min: {sub_v34[TARGET].min():.8f}, max: {sub_v34[TARGET].max():.8f}, unique: {sub_v34[TARGET].nunique():,}")
print(f"Spearman rank correlation with Taeyang 0.94650: {np.corrcoef(sub_v34[TARGET], r_taeyang)[0, 1]:.6f}")
print(f"Spearman rank correlation with pure shared 0.94649: {np.corrcoef(sub_v34[TARGET], r_shared)[0, 1]:.6f}")
print(f"Spearman rank correlation with Stacker: {np.corrcoef(sub_v34[TARGET], r_stacker)[0, 1]:.6f}")
print(f"Spearman rank correlation with Deotte Simpson: {np.corrcoef(sub_v34[TARGET], r_deotte)[0, 1]:.6f}")
