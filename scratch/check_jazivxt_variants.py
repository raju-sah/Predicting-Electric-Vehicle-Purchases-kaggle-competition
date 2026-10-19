import pandas as pd
import numpy as np
from scipy.stats import spearmanr

sub_main = pd.read_csv("scratch/kernels_pulled/jazivxt_zoom_out/submission.csv")["Will_Buy_EV"].values
sub_adj = pd.read_csv("scratch/kernels_pulled/jazivxt_zoom_out/submission_latest_adjusted.csv")["Will_Buy_EV"].values
sub_adj_mod = pd.read_csv("scratch/kernels_pulled/jazivxt_zoom_out/submission_latest_adjusted_model.csv")["Will_Buy_EV"].values
sub_taeyang = pd.read_csv("scratch/kernels_pulled/taeyangg4_094650_out/submission.csv")["Will_Buy_EV"].values

print("Spearman correlations:")
print(f"  main vs adj:       {spearmanr(sub_main, sub_adj)[0]:.7f}")
print(f"  main vs adj_mod:   {spearmanr(sub_main, sub_adj_mod)[0]:.7f}")
print(f"  adj vs taeyang:    {spearmanr(sub_adj, sub_taeyang)[0]:.7f}")
print(f"  main vs taeyang:   {spearmanr(sub_main, sub_taeyang)[0]:.7f}")
print(f"  adj_mod vs taeyang:{spearmanr(sub_adj_mod, sub_taeyang)[0]:.7f}")
