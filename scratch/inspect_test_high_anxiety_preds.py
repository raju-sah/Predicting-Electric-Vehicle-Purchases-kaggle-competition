import pandas as pd
import numpy as np

test = pd.read_csv("test.csv")
mask_test_zero = (test["Range_Anxiety_Level"] == "High") & ~((test["Environmental_Concern_Level"] == 5) & (test["Subsidy_Available"] == "Yes"))
print(f"Total test rows in High Anxiety Zero Zone: {mask_test_zero.sum()}")

files = {
    "v11_megayak": "subs/v11_megayak_4view_daily.csv",
    "v18_nina": "subs/v18_nina_honest_residual_tiebreaker.csv",
    "taeyang": "scratch/taeyangg4_out/submission.csv",
    "naji_xgb": "scratch/najiama_out/submission_XGBOOST.csv",
}

for name, path in files.items():
    df = pd.read_csv(path)
    col = [c for c in df.columns if c != "id"][0]
    preds = df.loc[mask_test_zero, col]
    print(f"\nModel {name}:")
    print(f"  Min prediction:    {preds.min():.6f}")
    print(f"  25% prediction:    {preds.quantile(0.25):.6f}")
    print(f"  Median prediction: {preds.median():.6f}")
    print(f"  75% prediction:    {preds.quantile(0.75):.6f}")
    print(f"  Max prediction:    {preds.max():.6f}")
    print(f"  Rows > 0.05:       {(preds > 0.05).sum()} / {len(preds)}")
    print(f"  Rows > 0.10:       {(preds > 0.10).sum()} / {len(preds)}")
