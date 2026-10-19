import pandas as pd
import numpy as np

sample = pd.read_csv("sample_submission.csv")
sub = pd.read_csv("scratch/kernels_pulled/taeyangg4_094650_out/submission.csv")

print("Checking submission integrity:")
print(f"Row count matches: {len(sub) == len(sample)} ({len(sub)} rows)")
print(f"Columns match: {list(sub.columns) == list(sample.columns)} ({list(sub.columns)})")
print(f"ID exact match: {(sub['id'] == sample['id']).all()}")
print(f"Null count: {sub['Will_Buy_EV'].isnull().sum()}")
print(f"Infinite count: {np.isinf(sub['Will_Buy_EV']).sum()}")
print(f"Min value: {sub['Will_Buy_EV'].min():.8f}")
print(f"Max value: {sub['Will_Buy_EV'].max():.8f}")
print(f"Unique values: {sub['Will_Buy_EV'].nunique():,}")
print(f"Zero ties: {sub['Will_Buy_EV'].nunique() == len(sub)}")

# Save to subs/v26_taeyang_lexsort_master.csv
sub.to_csv("subs/v26_taeyang_lexsort_master.csv", index=False)
print("Saved to subs/v26_taeyang_lexsort_master.csv")
