import os
import pandas as pd
import numpy as np
from scipy.stats import rankdata, spearmanr

print("--- BUILDING CANDIDATE V21 ---")

# 1. Load test features for boundary identification
test = pd.read_csv("test.csv")
sample = pd.read_csv("sample_submission.csv")
N = len(test)
assert N == 286571

# 2. Identify Boundary Masks
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

print(f"Test rows in Zero Cells:        {is_test_zero.sum():,} ({is_test_zero.mean():.4%})")
print(f"Test rows in Millionaire Cliff: {mask_test_millionaire.sum():,} ({mask_test_millionaire.mean():.4%})")

# 3. Load component predictions
p_anchor = pd.read_csv("subs/v18_nina_honest_residual_tiebreaker.csv")["Will_Buy_EV"].values
p_taeyang = pd.read_csv("scratch/taeyangg4_out/submission.csv")["Will_Buy_EV"].values
p_naji_xgb = pd.read_csv("scratch/najiama_out/submission_XGBOOST.csv")["Will_Buy_EV"].values
p_deotte_xgb = pd.read_csv("scratch/deotte_gpu_out/sub_xgb_deotte_simpson.csv")["Will_Buy_EV"].values
p_my_xgb = pd.read_csv("scratch/my_gpu_out/sub_xgb_fixed_digits.csv")["Will_Buy_EV"].values
p_naji_lgb = pd.read_csv("scratch/outputs/najiama_pure-lgbm-model-cv-0-94606-lb-0-94637/submission_LIGHTGBM.csv")["Will_Buy_EV"].values
p_realmlp = pd.read_csv("scratch/realmlp_out/submission.csv")["Will_Buy_EV"].values

# 4. Rank normalize each component
rk_anchor = rankdata(p_anchor, method="ordinal") / N
rk_taeyang = rankdata(p_taeyang, method="ordinal") / N
rk_naji_xgb = rankdata(p_naji_xgb, method="ordinal") / N
rk_deotte_xgb = rankdata(p_deotte_xgb, method="ordinal") / N
rk_my_xgb = rankdata(p_my_xgb, method="ordinal") / N
rk_naji_lgb = rankdata(p_naji_lgb, method="ordinal") / N
rk_realmlp = rankdata(p_realmlp, method="ordinal") / N

# 5. Construct 4-way Tree Blend
rk_tree_blend = 0.4312 * rk_naji_xgb + 0.2792 * rk_deotte_xgb + 0.1660 * rk_my_xgb + 0.1236 * rk_naji_lgb

# Combine anchor views (v18 + Taeyang both at 0.94645)
rk_top_anchor = 0.5 * rk_anchor + 0.5 * rk_taeyang

# We test 3 blending recipes:
# Recipe A (Conservative: 85% Anchor, 10% Tree Blend, 5% RealMLP)
# Recipe B (Balanced: 75% Anchor, 20% Tree Blend, 5% RealMLP)
# Recipe C (Aggressive: 65% Anchor, 30% Tree Blend, 5% RealMLP)

recipes = {
    "v21_recipe_A_cons": (0.85 * rk_top_anchor + 0.10 * rk_tree_blend + 0.05 * rk_realmlp),
    "v21_recipe_B_bal":  (0.75 * rk_top_anchor + 0.20 * rk_tree_blend + 0.05 * rk_realmlp),
    "v21_recipe_C_aggr": (0.65 * rk_top_anchor + 0.30 * rk_tree_blend + 0.05 * rk_realmlp),
}

for name, raw_score in recipes.items():
    score = raw_score.copy()
    
    # Apply deterministic boundary adjustments
    score[is_test_zero] = score[is_test_zero] * 0.05
    score[mask_test_millionaire] = score[mask_test_millionaire] + 10.0
    
    # Ordinal rank
    final_rank = rankdata(score, method="ordinal") / N
    
    # Check correlation with v11 and v18
    corr_v11 = spearmanr(final_rank, p_anchor).statistic
    print(f"\n{name}:")
    print(f"  Spearman correlation with v18 anchor: {corr_v11:.6f}")
    print(f"  Min: {final_rank.min():.8f}, Max: {final_rank.max():.8f}")
    print(f"  Unique values: {len(np.unique(final_rank)):,} / {N:,}")
    
    # Save submission file
    sub_df = pd.DataFrame({"id": sample["id"], "Will_Buy_EV": final_rank})
    out_path = f"subs/{name}.csv"
    sub_df.to_csv(out_path, index=False)
    print(f"  Saved {out_path}")
