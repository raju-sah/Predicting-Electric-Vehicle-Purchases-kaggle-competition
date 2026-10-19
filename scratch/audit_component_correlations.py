import pandas as pd
import numpy as np
from scipy.stats import spearmanr, rankdata

print("--- AUDITING TOP COMPONENTS AND CANDIDATES ---")

test = pd.read_csv("test.csv")
sample = pd.read_csv("sample_submission.csv")

components = {}

# 1. v18 Nina Honest Residual (LB 0.94645)
components["v18_nina"] = pd.read_csv("subs/v18_nina_honest_residual_tiebreaker.csv")["Will_Buy_EV"].values

# 2. v11 Megayak 4-view (LB 0.94645)
components["v11_megayak"] = pd.read_csv("subs/v11_megayak_4view_daily.csv")["Will_Buy_EV"].values

# 3. Taeyang 0.94650 Lexsort Master (LB 0.94650)
components["taeyang_094650"] = pd.read_csv("scratch/kernels_pulled/taeyangg4_094650_out/submission.csv")["Will_Buy_EV"].values

# 4. Jazivxt Zoom-Zoom (LB 0.94649+)
components["jazivxt_zoom"] = pd.read_csv("scratch/kernels_pulled/jazivxt_zoom_out/submission.csv")["Will_Buy_EV"].values

# 5. Jazivxt Adjusted
components["jazivxt_adj"] = pd.read_csv("scratch/kernels_pulled/jazivxt_zoom_out/submission_latest_adjusted.csv")["Will_Buy_EV"].values

# 6. RealMLP PyTorch Continuous
components["realmlp"] = pd.read_csv("scratch/realmlp_out/submission.csv")["Will_Buy_EV"].values

# 7. 10-Model Logistic Stacker (OOF 0.946295)
components["meta_stacker_10way"] = np.load("scratch/test_meta_stacker_10way.npy")

# 8. Deotte Simpson GPU XGBoost
components["deotte_simpson"] = pd.read_csv("scratch/deotte_gpu_out/sub_xgb_deotte_simpson.csv")["Will_Buy_EV"].values

# 9. Najiama Triple-TE XGBoost (LB 0.94639)
components["naji_xgboost"] = pd.read_csv("scratch/najiama_out/submission_XGBOOST.csv")["Will_Buy_EV"].values

names = list(components.keys())
print(f"Loaded {len(names)} top components:")

for name in names:
    arr = components[name]
    ties = len(arr) - len(np.unique(arr))
    print(f"  {name:20s}: min={arr.min():.5f}, max={arr.max():.5f}, mean={arr.mean():.5f}, unique={len(np.unique(arr)):,}, ties={ties:,}")

# Spearman rank correlation matrix
corr_matrix = np.zeros((len(names), len(names)))
for i in range(len(names)):
    for j in range(len(names)):
        corr_matrix[i, j] = spearmanr(components[names[i]], components[names[j]])[0]

corr_df = pd.DataFrame(corr_matrix, index=names, columns=names)
print("\n--- SPEARMAN RANK CORRELATION MATRIX ---")
print(corr_df.round(5))
