import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

print("--- PURE 10-MODEL META-STACKER SUBMISSION ---")

sample = pd.read_csv("sample_submission.csv")
test = pd.read_csv("test.csv")
N = len(test)

stacker_raw = np.load("scratch/test_meta_stacker_10way.npy")
v11 = pd.read_csv("subs/v11_megayak_4view_daily.csv")["Will_Buy_EV"].values

# 1. Check boundary conditions on stacker
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

# Pure stacker with boundary calibration
s1 = stacker_raw.copy()
s1[is_test_zero] = s1[is_test_zero] * 0.05
s1[mask_test_millionaire] = s1[mask_test_millionaire] + 10.0
rank_pure_stacker = rankdata(s1, method="ordinal") / N

print(f"Pure Stacker vs v11 Spearman correlation: {spearmanr(rank_pure_stacker, v11).statistic:.6f}")

# 50/50 Stacker + Anchor
rk_v11 = rankdata(v11, method="ordinal") / N
s2 = 0.50 * rk_v11 + 0.50 * (rankdata(stacker_raw, method="ordinal") / N)
s2[is_test_zero] = s2[is_test_zero] * 0.05
s2[mask_test_millionaire] = s2[mask_test_millionaire] + 10.0
rank_50_50 = rankdata(s2, method="ordinal") / N
print(f"50/50 Stacker+Anchor vs v11 Spearman correlation: {spearmanr(rank_50_50, v11).statistic:.6f}")

# Save both
sub_pure = pd.DataFrame({"id": sample["id"], "Will_Buy_EV": rank_pure_stacker})
sub_pure.to_csv("subs/v23_pure_10model_meta_stacker.csv", index=False)
print("Saved subs/v23_pure_10model_meta_stacker.csv")

sub_5050 = pd.DataFrame({"id": sample["id"], "Will_Buy_EV": rank_50_50})
sub_5050.to_csv("subs/v24_meta_stack_50_50_balanced.csv", index=False)
print("Saved subs/v24_meta_stack_50_50_balanced.csv")
