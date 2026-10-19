import pandas as pd
import numpy as np
from scipy.stats import spearmanr

p_jazi = pd.read_csv("scratch/kernels_pulled/jazivxt_zoom_out/submission.csv")["Will_Buy_EV"].values
p_taeyang = pd.read_csv("scratch/kernels_pulled/taeyangg4_094650_out/submission.csv")["Will_Buy_EV"].values

corr = spearmanr(p_jazi, p_taeyang)[0]
print(f"Spearman rank correlation between Jazivxt and Taeyang: {corr:.7f}")

diff = np.abs(p_jazi - p_taeyang)
print(f"Mean abs diff: {diff.mean():.6f}, max: {diff.max():.6f}")
