import pandas as pd
import numpy as np
from scipy.stats import spearmanr, rankdata

files = {
    "v11_megayak": "subs/v11_megayak_4view_daily.csv",
    "taeyang": "scratch/taeyangg4_out/submission.csv",
    "v18_nina": "subs/v18_nina_honest_residual_tiebreaker.csv",
    "naji_xgb": "scratch/najiama_out/submission_XGBOOST.csv",
    "deotte_xgb": "scratch/deotte_gpu_out/sub_xgb_deotte_simpson.csv",
    "naji_lgb": "scratch/outputs/najiama_pure-lgbm-model-cv-0-94637/submission_LIGHTGBM.csv" if False else "scratch/outputs/najiama_pure-lgbm-model-cv-0-94606-lb-0-94637/submission_LIGHTGBM.csv",
    "my_xgb": "scratch/my_gpu_out/sub_xgb_fixed_digits.csv",
    "realmlp": "scratch/realmlp_out/submission.csv",
}

sample = pd.read_csv("sample_submission.csv")
preds = {}

for name, path in files.items():
    df = pd.read_csv(path)
    assert len(df) == len(sample)
    assert (df["id"] == sample["id"]).all()
    col = [c for c in df.columns if c != "id"][0]
    preds[name] = df[col].values
    print(f"Loaded {name:15s} from {path:65s}: min={preds[name].min():.6f}, max={preds[name].max():.6f}, std={preds[name].std():.6f}")

names = list(preds.keys())
corr_mat = np.zeros((len(names), len(names)))
for i in range(len(names)):
    for j in range(len(names)):
        corr_mat[i, j] = spearmanr(preds[names[i]], preds[names[j]]).statistic

corr_df = pd.DataFrame(corr_mat, index=names, columns=names)
print("\nTest Spearman Rank Correlation Matrix:")
print(corr_df.round(5))
