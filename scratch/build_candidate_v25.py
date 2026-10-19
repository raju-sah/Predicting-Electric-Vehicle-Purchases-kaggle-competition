import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

print("--- BUILDING CANDIDATE V25: GRANDMASTER MULTI-PARADIGM BLEND ---")

sample = pd.read_csv("sample_submission.csv")
test = pd.read_csv("test.csv")
N = len(test)

# 1. Load components
v18 = pd.read_csv("subs/v18_nina_honest_residual_tiebreaker.csv")["Will_Buy_EV"].values
taeyang = pd.read_csv("scratch/taeyangg4_out/submission.csv")["Will_Buy_EV"].values
v11 = pd.read_csv("subs/v11_megayak_4view_daily.csv")["Will_Buy_EV"].values
stacker_10 = np.load("scratch/test_meta_stacker_10way.npy")
realmlp = pd.read_csv("scratch/realmlp_out/submission.csv")["Will_Buy_EV"].values
deotte_xgb = pd.read_csv("scratch/deotte_gpu_out/sub_xgb_deotte_simpson.csv")["Will_Buy_EV"].values
naji_xgb = pd.read_csv("scratch/najiama_out/submission_XGBOOST.csv")["Will_Buy_EV"].values

# 2. Ordinal rank transform each component to [0, 1]
rk_v18 = rankdata(v18, method="ordinal") / N
rk_taeyang = rankdata(taeyang, method="ordinal") / N
rk_v11 = rankdata(v11, method="ordinal") / N
rk_stacker = rankdata(stacker_10, method="ordinal") / N
rk_realmlp = rankdata(realmlp, method="ordinal") / N
rk_deotte = rankdata(deotte_xgb, method="ordinal") / N
rk_naji = rankdata(naji_xgb, method="ordinal") / N

# 3. Super Anchor (blending v18 + taeyang + v11)
rk_super_anchor = (rk_v18 + rk_taeyang + rk_v11) / 3.0

# 4. We formulate candidate v25:
# 78% Super Anchor
# 12% 10-Model Logistic Stacker
# 5% RealMLP Neural Network
# 5% Deotte Simpson XGBoost
blend_v25 = (
    0.78 * rk_super_anchor +
    0.12 * rk_stacker +
    0.05 * rk_realmlp +
    0.05 * rk_deotte
)

# 5. Deterministic Boundary Constraints
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

# Apply boundaries
score_v25 = blend_v25.copy()
score_v25[is_test_zero] = score_v25[is_test_zero] * 0.05
score_v25[mask_test_millionaire] = score_v25[mask_test_millionaire] + 10.0

# 6. Apply Nina's 0.74 power-law residual tie-breaker
diff = score_v25 - np.round(score_v25, 6)
residual = np.sign(diff) * (np.abs(diff) ** 0.74)
score_v25_nina = score_v25 + residual * 1e-6

# Final tie-free ordinal ranking
final_v25 = rankdata(score_v25_nina, method="ordinal") / N

print(f"\nCandidate v25 Diagnostics:")
print(f"  Spearman correlation with v18 anchor: {spearmanr(final_v25, v18).statistic:.6f}")
print(f"  Spearman correlation with v11 anchor: {spearmanr(final_v25, v11).statistic:.6f}")
print(f"  Spearman correlation with stacker:    {spearmanr(final_v25, stacker_10).statistic:.6f}")
print(f"  Spearman correlation with RealMLP:    {spearmanr(final_v25, realmlp).statistic:.6f}")
print(f"  Min rank: {final_v25.min():.8f}, Max rank: {final_v25.max():.8f}")
print(f"  Unique values: {len(np.unique(final_v25)):,} / {N:,}")

out_path = "subs/v25_super_anchor_stacker_realmlp_nina.csv"
sub_df = pd.DataFrame({"id": sample["id"], "Will_Buy_EV": final_v25})
sub_df.to_csv(out_path, index=False)
print(f"Saved {out_path}")

# Integrity checks
assert list(sub_df.columns) == ["id", "Will_Buy_EV"]
assert len(sub_df) == 286571
assert (sub_df["id"] == sample["id"]).all()
assert not sub_df["Will_Buy_EV"].isna().any()
assert sub_df["Will_Buy_EV"].nunique() == 286571
print("INTEGRITY VERIFIED: 100% clean submission ready for Kaggle.")
