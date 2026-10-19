import pandas as pd
import numpy as np
from scipy.stats import spearmanr

sub_main = pd.read_csv("scratch/kernels_pulled/jazivxt_zoom_sep14/submission.csv")
sub_hist = pd.read_csv("scratch/kernels_pulled/jazivxt_zoom_sep14/submission_history_adjusted.csv")
sub_v26 = pd.read_csv("subs/v26_taeyang_lexsort_master.csv")

print("Jazivxt Sep 14 sub_main:")
print(f"  Shape: {sub_main.shape}, unique: {sub_main['Will_Buy_EV'].nunique():,}, min: {sub_main['Will_Buy_EV'].min():.6f}, max: {sub_main['Will_Buy_EV'].max():.6f}")

print("\nJazivxt Sep 14 sub_history_adjusted:")
print(f"  Shape: {sub_hist.shape}, unique: {sub_hist['Will_Buy_EV'].nunique():,}, min: {sub_hist['Will_Buy_EV'].min():.6f}, max: {sub_hist['Will_Buy_EV'].max():.6f}")

print("\nSpearman correlations:")
print(f"  sub_main vs sub_v26: {spearmanr(sub_main['Will_Buy_EV'], sub_v26['Will_Buy_EV'])[0]:.7f}")
print(f"  sub_hist vs sub_v26: {spearmanr(sub_hist['Will_Buy_EV'], sub_v26['Will_Buy_EV'])[0]:.7f}")
print(f"  sub_main vs sub_hist:{spearmanr(sub_main['Will_Buy_EV'], sub_hist['Will_Buy_EV'])[0]:.7f}")
