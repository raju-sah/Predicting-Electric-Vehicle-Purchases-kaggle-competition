import pandas as pd
import numpy as np

f1 = pd.read_csv("scratch/kernels_pulled/taeyangg4_094650_out/submission.csv")
f2 = pd.read_csv("subs/v26_taeyang_lexsort_master.csv")

diff_order = (np.argsort(f1["Will_Buy_EV"]) != np.argsort(f2["Will_Buy_EV"])).sum()
print(f"Number of rows with different sorting order between f1 and f2: {diff_order:,}")
print(f"Unique in f1: {f1['Will_Buy_EV'].nunique():,}")
print(f"Unique in f2: {f2['Will_Buy_EV'].nunique():,}")
