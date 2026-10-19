import pandas as pd
import numpy as np
from scipy.stats import spearmanr

sub_lgb = pd.read_csv("scratch/kernels_pulled/jazivxt_zoom_sep14/submission_lightgbm.csv")["Will_Buy_EV"].values
sub_v26 = pd.read_csv("subs/v26_taeyang_lexsort_master.csv")["Will_Buy_EV"].values
sub_v18 = pd.read_csv("subs/v18_nina_honest_residual_tiebreaker.csv")["Will_Buy_EV"].values

print("submission_lightgbm properties:")
print(f"  Unique values: {len(np.unique(sub_lgb)):,}, ties: {len(sub_lgb) - len(np.unique(sub_lgb)):,}")
print(f"  Spearman correlation with v26 (0.94649): {spearmanr(sub_lgb, sub_v26)[0]:.6f}")
print(f"  Spearman correlation with v18 (0.94645): {spearmanr(sub_lgb, sub_v18)[0]:.6f}")
