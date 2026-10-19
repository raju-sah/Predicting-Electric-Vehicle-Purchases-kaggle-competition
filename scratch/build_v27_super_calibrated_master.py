import pandas as pd
import numpy as np
from scipy.stats import rankdata

print("--- BUILDING V27: COMPLETE ZERO-CELL CALIBRATED GRANDMASTER ---")

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sample = pd.read_csv("sample_submission.csv")
TARGET = "Will_Buy_EV"
ID = "id"

def rk01(arr: np.ndarray) -> np.ndarray:
    return (rankdata(arr, method="ordinal") - 0.5) / len(arr)

def lexsort_ranking(primary: np.ndarray, secondary: np.ndarray) -> np.ndarray:
    order = np.lexsort((secondary, primary))
    ranks = np.empty(len(order), dtype=np.int64)
    ranks[order] = np.arange(1, len(order) + 1)
    return (ranks - 0.5) / len(ranks)

# 1. Load components
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

# 2. Complete 10 zero-cell mask
train["cell"] = (
    train["Environmental_Concern_Level"].astype(str)
    + "_"
    + train["Subsidy_Available"].astype(str)
    + "_"
    + train["Range_Anxiety_Level"].astype(str)
)
test["cell"] = (
    test["Environmental_Concern_Level"].astype(str)
    + "_"
    + test["Subsidy_Available"].astype(str)
    + "_"
    + test["Range_Anxiety_Level"].astype(str)
)
cell_stats = train.groupby("cell")["Will_Buy_EV"].agg(
    count="count", pos=lambda s: (s == "Yes").sum()
)
zero_cells = cell_stats[cell_stats["pos"] == 0].index.tolist()
mask_10_zero_cells = test["cell"].isin(zero_cells).values
print(f"Test rows in complete 10 zero cells: {mask_10_zero_cells.sum():,}")

# Domain boundary masks:
income = pd.to_numeric(test["Annual_Income_USD"], errors="coerce").to_numpy()
commute = pd.to_numeric(test["Daily_Commute_km"], errors="coerce").to_numpy()

mask_upper_cliff = (income >= 170537.0)
mask_dead_inc = (income >= 31004.0) & (income <= 41970.0)
mask_commute = (commute >= 83.0)

print(f"Upper Cliff Rows (100% EV buyers): {mask_upper_cliff.sum():,}")
print(f"Dead Zone Income Rows:              {mask_dead_inc.sum():,}")
print(f"Commute Dead Zone Rows:             {mask_commute.sum():,}")

# 3. Super Blend formulation:
# 88% v26 (Rank 20 champion) + 5% 10-Model Stacker + 4% v18 Nina + 2% Deotte Simpson + 1% RealMLP
blend = (
    0.88 * r_v26
    + 0.05 * r_stacker
    + 0.04 * r_v18
    + 0.02 * r_deotte
    + 0.01 * r_rmlp
)

# Apply Boundary Calibrations:
# For Millionaire Cliff: boost to extreme top
blend[mask_upper_cliff] += 20.0

# For Complete 10 Zero Cells: drop firmly down to zero-buyer floor
blend[mask_10_zero_cells] -= 10.0

# Dead income and extreme commute
blend[mask_dead_inc] -= 5.0
blend[mask_commute] -= 3.0

# Final lexicographical tie-breaking using RealMLP continuous probabilities
final_ranks = lexsort_ranking(blend, p_rmlp)

sub_v27 = pd.DataFrame({
    ID: test[ID].to_numpy(),
    TARGET: final_ranks,
})

# Verifications
assert len(sub_v27) == len(sample), "Row count mismatch!"
assert sub_v27[ID].equals(sample[ID]), "ID mismatch!"
assert not sub_v27[TARGET].isnull().any(), "Null values detected!"
assert sub_v27[TARGET].nunique() == len(sample), "Ties detected!"
assert sub_v27[TARGET].min() >= 0.0 and sub_v27[TARGET].max() <= 1.0, "Out of bounds!"

sub_v27.to_csv("subs/v27_complete_zero_calibrated_master.csv", index=False)
print("SUCCESS: subs/v27_complete_zero_calibrated_master.csv created successfully!")
print(f"Ranks min: {sub_v27[TARGET].min():.8f}, max: {sub_v27[TARGET].max():.8f}, unique: {sub_v27[TARGET].nunique():,}")
print(f"Spearman rank correlation with v26 (Rank 20): {np.corrcoef(sub_v27[TARGET], r_v26)[0, 1]:.6f}")
