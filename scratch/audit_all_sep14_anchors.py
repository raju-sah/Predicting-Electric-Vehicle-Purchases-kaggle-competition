import pandas as pd
import numpy as np
from scipy.stats import spearmanr

print("--- AUDITING ALL CANDIDATE ANCHORS ON SEP 14 ---")

anchors = {}

# 1. Jazivxt Sep 14 pass-2
anchors["jazi_sep14"] = pd.read_csv("scratch/kernels_pulled/jazivxt_zoom_sep14/submission.csv")["Will_Buy_EV"].values

# 2. Taeyang Lexsort Master
anchors["taeyang_lexsort"] = pd.read_csv("scratch/kernels_pulled/taeyangg4_094650_out/submission.csv")["Will_Buy_EV"].values

# 3. Nina h-blend 4
anchors["nina_h4"] = pd.read_csv("scratch/kernels_pulled/nina_h4_out/submission.csv")["Will_Buy_EV"].values

# 4. Souke Aizen v2
anchors["souke_v2"] = pd.read_csv("scratch/kernels_pulled/souke_v2_out/submission.csv")["Will_Buy_EV"].values

# 5. v18 Nina Honest Residual (our best 0.94645 anchor)
anchors["v18_nina"] = pd.read_csv("subs/v18_nina_honest_residual_tiebreaker.csv")["Will_Buy_EV"].values

# 6. Our 10-Model Stacker
anchors["meta_stacker"] = np.load("scratch/test_meta_stacker_10way.npy")

# 7. RealMLP PyTorch
anchors["realmlp"] = pd.read_csv("scratch/realmlp_out/submission.csv")["Will_Buy_EV"].values

# 8. Deotte Simpson GPU XGB
anchors["deotte_simpson"] = pd.read_csv("scratch/deotte_gpu_out/sub_xgb_deotte_simpson.csv")["Will_Buy_EV"].values

names = list(anchors.keys())
print(f"Loaded {len(names)} candidate components:")

for name in names:
    arr = anchors[name]
    ties = len(arr) - len(np.unique(arr))
    print(f"  {name:18s}: min={arr.min():.6f}, max={arr.max():.6f}, mean={arr.mean():.6f}, unique={len(np.unique(arr)):,}, ties={ties:,}")

# Fast Spearman correlation using pre-ranked arrays
from scipy.stats import rankdata
N = len(anchors["jazi_sep14"])
ranks = {name: rankdata(anchors[name], method="ordinal") / N for name in names}

corr_matrix = pd.DataFrame(index=names, columns=names, dtype=float)
for n1 in names:
    for n2 in names:
        corr_matrix.loc[n1, n2] = np.corrcoef(ranks[n1], ranks[n2])[0, 1]

print("\n--- SPEARMAN RANK CORRELATION MATRIX ---")
print(corr_matrix.round(5))
