import pandas as pd
import numpy as np

print("--- AUDITING TAEYANG'S ZERO CELLS VS COMPLETE 10 ZERO CELLS ---")

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
y_train = (train["Will_Buy_EV"] == "Yes").astype(int).values

# 1. Complete 10 zero cells
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
    count="count",
    pos=lambda s: (s == "Yes").sum(),
    rate=lambda s: (s == "Yes").mean()
).reset_index()

zero_cells = cell_stats[cell_stats["pos"] == 0]["cell"].tolist()
print(f"Total zero cells in train: {len(zero_cells)}")
print(cell_stats[cell_stats["pos"] == 0])

mask_complete_zero_test = test["cell"].isin(zero_cells).values
print(f"\nTest rows in complete 10 zero cells: {mask_complete_zero_test.sum():,}")

# 2. Taeyang's mask:
income_test = pd.to_numeric(test["Annual_Income_USD"], errors="coerce").values
commute_test = pd.to_numeric(test["Daily_Commute_km"], errors="coerce").values
subsidy_no_test = (test["Subsidy_Available"].astype(str) == "No").values
env_1_test = (pd.to_numeric(test["Environmental_Concern_Level"], errors="coerce") == 1.0).values
anx_med_high_test = test["Range_Anxiety_Level"].isin(["Medium", "High"]).values

taeyang_30k_zero = (income_test == 30000.0) & subsidy_no_test & (env_1_test | anx_med_high_test)
print(f"Taeyang 30k zero cell rows: {taeyang_30k_zero.sum():,}")

# Difference: rows in our zero cells that Taeyang MISSED
missed_by_taeyang = mask_complete_zero_test & (~taeyang_30k_zero)
print(f"Zero-cell rows missed by Taeyang: {missed_by_taeyang.sum():,}")

# What rank did Taeyang give to these missed zero-cell rows?
sub_taeyang = pd.read_csv("subs/v26_taeyang_lexsort_master.csv")
missed_ranks = sub_taeyang.loc[missed_by_taeyang, "Will_Buy_EV"].values
print(f"Taeyang's predictions on missed zero rows:")
print(f"  Min rank:    {missed_ranks.min():.5f}")
print(f"  Median rank: {np.median(missed_ranks):.5f}")
print(f"  Max rank:    {missed_ranks.max():.5f} (Should be near 0.00000!)")
print(f"  Rows with rank > 0.10: {(missed_ranks > 0.10).sum():,} / {len(missed_ranks):,}")
print(f"  Rows with rank > 0.30: {(missed_ranks > 0.30).sum():,} / {len(missed_ranks):,}")
