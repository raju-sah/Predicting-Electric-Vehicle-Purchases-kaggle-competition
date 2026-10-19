import pandas as pd
import numpy as np

sample = pd.read_csv("sample_submission.csv")
sub = pd.read_csv("scratch/kernels_pulled/jazivxt_zoom_sep14/submission.csv")

print("--- VERIFYING V30 SUBMISSION ---")
print(f"Row count matches: {len(sub) == len(sample)} ({len(sub)} rows)")
print(f"Columns match: {list(sub.columns) == list(sample.columns)} ({list(sub.columns)})")
print(f"ID exact match: {(sub['id'] == sample['id']).all()}")
print(f"Null count: {sub['Will_Buy_EV'].isnull().sum()}")
print(f"Infinite count: {np.isinf(sub['Will_Buy_EV']).sum()}")
print(f"Min value: {sub['Will_Buy_EV'].min():.8f}")
print(f"Max value: {sub['Will_Buy_EV'].max():.8f}")
print(f"Unique values: {sub['Will_Buy_EV'].nunique():,}")
print(f"Zero ties: {sub['Will_Buy_EV'].nunique() == len(sub)}")

# Save to subs/v30_jazivxt_sep14_pass2_baseline.csv
sub.to_csv("subs/v30_jazivxt_sep14_pass2_baseline.csv", index=False)
print("Saved to subs/v30_jazivxt_sep14_pass2_baseline.csv")
