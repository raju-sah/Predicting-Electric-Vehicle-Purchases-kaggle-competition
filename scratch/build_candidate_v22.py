import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

print("--- BUILDING CANDIDATE V22: 10-MODEL META-STACK BLEND ---")

test = pd.read_csv("test.csv")
sample = pd.read_csv("sample_submission.csv")
N = len(test)

# 1. Load components
v18 = pd.read_csv("subs/v18_nina_honest_residual_tiebreaker.csv")["Will_Buy_EV"].values
taeyang = pd.read_csv("scratch/taeyangg4_out/submission.csv")["Will_Buy_EV"].values
stacker = np.load("scratch/test_meta_stacker_10way.npy")
realmlp = pd.read_csv("scratch/realmlp_out/submission.csv")["Will_Buy_EV"].values

# 2. Rank normalize each
rk_anchor = 0.5 * (rankdata(v18, method="ordinal") / N) + 0.5 * (rankdata(taeyang, method="ordinal") / N)
rk_stacker = rankdata(stacker, method="ordinal") / N
rk_realmlp = rankdata(realmlp, method="ordinal") / N

# 3. Recipes
# Recipe 1: 70% Anchor + 25% Stacker + 5% RealMLP
raw_v22_A = 0.70 * rk_anchor + 0.25 * rk_stacker + 0.05 * rk_realmlp
# Recipe 2: 55% Anchor + 40% Stacker + 5% RealMLP
raw_v22_B = 0.55 * rk_anchor + 0.40 * rk_stacker + 0.05 * rk_realmlp

# 4. Boundary masks
zero_cell_tuples = [
    (1.0, "No", "High"),
    (1.0, "Yes", "High"),
    (2.0, "No", "High"),
    (2.0, "No", "Medium"),
    (2.0, "Yes", "High"),
    (3.0, "No", "High"),
    (3.0, "Yes", "High"),
    (4.0, "No", "High"),
    (4.0, "Yes", "High"),
    (5.0, "No", "High"),
]

is_test_zero = pd.Series(False, index=test.index)
for c, s, a in zero_cell_tuples:
    m = (test["Environmental_Concern_Level"] == c) & (test["Subsidy_Available"] == s) & (test["Range_Anxiety_Level"] == a)
    is_test_zero |= m

mask_test_millionaire = (test["Annual_Income_USD"] >= 170537)

for name, raw in [("v22_meta_stack_70_25_5", raw_v22_A), ("v22_meta_stack_55_40_5", raw_v22_B)]:
    score = raw.copy()
    score[is_test_zero] = score[is_test_zero] * 0.05
    score[mask_test_millionaire] = score[mask_test_millionaire] + 10.0
    
    final_rank = rankdata(score, method="ordinal") / N
    
    corr_v18 = spearmanr(final_rank, v18).statistic
    print(f"\n{name}:")
    print(f"  Spearman correlation with v18: {corr_v18:.6f}")
    print(f"  Min: {final_rank.min():.8f}, Max: {final_rank.max():.8f}")
    print(f"  Unique values: {len(np.unique(final_rank)):,} / {N:,}")
    
    out_path = f"subs/{name}.csv"
    sub_df = pd.DataFrame({"id": sample["id"], "Will_Buy_EV": final_rank})
    sub_df.to_csv(out_path, index=False)
    print(f"  Saved {out_path}")
    
    # Verify integrity
    assert list(sub_df.columns) == ["id", "Will_Buy_EV"]
    assert len(sub_df) == 286571
    assert (sub_df["id"] == sample["id"]).all()
    assert not sub_df["Will_Buy_EV"].isna().any()
    assert sub_df["Will_Buy_EV"].nunique() == 286571
    print(f"  VERIFICATION PASSED: 100% clean, zero ties, valid bounds.")
