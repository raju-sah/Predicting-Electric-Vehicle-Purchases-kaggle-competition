import pandas as pd
import numpy as np

for fname in ["v21_recipe_A_cons.csv", "v21_recipe_B_bal.csv", "v21_recipe_C_aggr.csv"]:
    path = f"subs/{fname}"
    sub = pd.read_csv(path)
    sample = pd.read_csv("sample_submission.csv")
    
    assert list(sub.columns) == ["id", "Will_Buy_EV"], f"Header mismatch in {fname}"
    assert len(sub) == len(sample) == 286571, f"Row count mismatch in {fname}"
    assert (sub["id"] == sample["id"]).all(), f"ID mismatch in {fname}"
    assert not sub["Will_Buy_EV"].isna().any(), f"NaN detected in {fname}"
    assert not np.isinf(sub["Will_Buy_EV"]).any(), f"Inf detected in {fname}"
    assert sub["Will_Buy_EV"].min() >= 0.0 and sub["Will_Buy_EV"].max() <= 1.0, f"Range error in {fname}"
    assert sub["Will_Buy_EV"].nunique() == 286571, f"Ties detected in {fname}"
    print(f"VERIFIED {fname}: 100% CLEAN! Zero ties, exact format, valid ranges.")
