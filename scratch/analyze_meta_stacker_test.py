import numpy as np
import pandas as pd
from scipy.stats import spearmanr, rankdata

test_stack = np.load("scratch/test_meta_stacker_10way.npy")
v18 = pd.read_csv("subs/v18_nina_honest_residual_tiebreaker.csv")["Will_Buy_EV"].values
v11 = pd.read_csv("subs/v11_megayak_4view_daily.csv")["Will_Buy_EV"].values
taeyang = pd.read_csv("scratch/taeyangg4_out/submission.csv")["Will_Buy_EV"].values
realmlp = pd.read_csv("scratch/realmlp_out/submission.csv")["Will_Buy_EV"].values

print(f"Meta-stacker test shape: {test_stack.shape}")
print(f"Meta-stacker range: min={test_stack.min():.6f}, max={test_stack.max():.6f}, mean={test_stack.mean():.6f}")

print("\nSpearman Rank Correlations:")
print(f"  vs v18 anchor:  {spearmanr(test_stack, v18).statistic:.6f}")
print(f"  vs v11 anchor:  {spearmanr(test_stack, v11).statistic:.6f}")
print(f"  vs taeyang:     {spearmanr(test_stack, taeyang).statistic:.6f}")
print(f"  vs realmlp:     {spearmanr(test_stack, realmlp).statistic:.6f}")
