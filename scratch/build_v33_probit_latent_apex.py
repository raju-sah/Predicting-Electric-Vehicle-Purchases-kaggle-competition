import pandas as pd
import numpy as np
from scipy.stats import rankdata, norm

print("--- BUILDING V33: LATENT PROBIT-SPACE OPTIMAL APEX ---")

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

# Normalized ordinal rank representations
r_shared = rk01(shared_094649)
r_taeyang = rk01(p_taeyang)
r_deotte = rk01(p_deotte)
r_stacker = rk01(p_stacker)
r_rmlp = rk01(p_rmlp)

# 2. Map to Probit Latent Space (Inverse Normal CDF)
eps = 1e-7
z_shared = norm.ppf(np.clip(r_shared, eps, 1.0 - eps))
z_taeyang = norm.ppf(np.clip(r_taeyang, eps, 1.0 - eps))
z_deotte = norm.ppf(np.clip(r_deotte, eps, 1.0 - eps))
z_stacker = norm.ppf(np.clip(r_stacker, eps, 1.0 - eps))
z_rmlp = norm.ppf(np.clip(r_rmlp, eps, 1.0 - eps))

print("Probit latent z stats:")
print(f"  z_shared: min={z_shared.min():.4f}, max={z_shared.max():.4f}, mean={z_shared.mean():.4f}")
print(f"  z_deotte: min={z_deotte.min():.4f}, max={z_deotte.max():.4f}, mean={z_deotte.mean():.4f}")

# 3. Combine in Probit Latent Space:
# 82% pure shared 0.94649 + 8% Taeyang 0.94650 + 4% Deotte Simpson GPU XGB + 3% 10-Model Stacker + 3% RealMLP
z_blend = (
    0.82 * z_shared
    + 0.08 * z_taeyang
    + 0.04 * z_deotte
    + 0.03 * z_stacker
    + 0.03 * z_rmlp
)

# Convert back to probability space
p_blend = norm.cdf(z_blend)

# 4. Domain boundary masks
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
p_blend[mask_upper] += 10.0
p_blend[mask_dead_inc] -= 10.0
p_blend[mask_commute] -= 5.0
p_blend[mask_30k_zero] -= 5.0

# Hierarchical Lexicographical Sorting via continuous RealMLP
final_ranks = lexsort_ranking(p_blend, p_rmlp)

sub_v33 = pd.DataFrame({
    ID: test[ID].to_numpy(),
    TARGET: final_ranks,
})

# Strict Verifications
assert len(sub_v33) == len(sample), "Row count mismatch!"
assert sub_v33[ID].equals(sample[ID]), "ID mismatch!"
assert not sub_v33[TARGET].isnull().any(), "Null values detected!"
assert sub_v33[TARGET].nunique() == len(sample), "Ties detected!"
assert sub_v33[TARGET].min() >= 0.0 and sub_v33[TARGET].max() <= 1.0, "Out of bounds!"

sub_v33.to_csv("subs/v33_probit_latent_apex.csv", index=False)
print("SUCCESS: subs/v33_probit_latent_apex.csv created successfully!")
print(f"Ranks min: {sub_v33[TARGET].min():.8f}, max: {sub_v33[TARGET].max():.8f}, unique: {sub_v33[TARGET].nunique():,}")
print(f"Spearman rank correlation with Taeyang 0.94650: {np.corrcoef(sub_v33[TARGET], r_taeyang)[0, 1]:.6f}")
print(f"Spearman rank correlation with pure shared 0.94649: {np.corrcoef(sub_v33[TARGET], r_shared)[0, 1]:.6f}")
print(f"Spearman rank correlation with Deotte Simpson: {np.corrcoef(sub_v33[TARGET], r_deotte)[0, 1]:.6f}")
