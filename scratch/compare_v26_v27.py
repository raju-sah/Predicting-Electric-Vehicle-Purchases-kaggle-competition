import pandas as pd
import numpy as np

sub26 = pd.read_csv("subs/v26_taeyang_lexsort_master.csv")
sub27 = pd.read_csv("subs/v27_complete_zero_calibrated_master.csv")

diff = np.abs(sub26["Will_Buy_EV"] - sub27["Will_Buy_EV"])
print(f"Max absolute rank difference: {diff.max():.6f}")
print(f"Mean absolute rank difference: {diff.mean():.6f}")
print(f"Number of rows with rank changed > 0.001: {(diff > 0.001).sum():,}")
print(f"Number of rows with rank changed > 0.01: {(diff > 0.01).sum():,}")
print(f"Number of rows with rank changed > 0.05: {(diff > 0.05).sum():,}")

# Where did large rank changes happen?
test = pd.read_csv("test.csv")
test["diff"] = diff
large_diffs = test[test["diff"] > 0.05]
print("\nSample rows with large rank differences (> 0.05):")
print(large_diffs[["id", "Annual_Income_USD", "Environmental_Concern_Level", "Subsidy_Available", "Range_Anxiety_Level", "diff"]].head(10))
