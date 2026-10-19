"""Validate v71 and v72 submissions before Kaggle upload."""
import pandas as pd
import numpy as np

BASE = "/home/raju/AI-ML Projects/Kaggle Competition/Predicting Electric Vehicle Purchases"
sample = pd.read_csv(f"{BASE}/sample_submission.csv")

for name, fname in [("v71", "v71_megadiv_bandswap.csv"), ("v72", "v72_extended_6band_swap.csv")]:
    sub = pd.read_csv(f"{BASE}/subs/{fname}")
    
    # Check header
    assert sub.columns.tolist() == ["id", "Will_Buy_EV"], f"{name}: wrong columns: {sub.columns.tolist()}"
    
    # Check IDs match
    assert (sub["id"] == sample["id"]).all(), f"{name}: ID mismatch"
    
    # Check values
    assert sub["Will_Buy_EV"].notna().all(), f"{name}: has NaN values"
    assert np.isfinite(sub["Will_Buy_EV"]).all(), f"{name}: has infinite values"
    assert len(sub) == len(sample), f"{name}: row count mismatch"
    
    print(f"{name} ({fname}): VALID")
    print(f"  Rows: {len(sub)}")
    print(f"  Pred range: [{sub['Will_Buy_EV'].min():.6f}, {sub['Will_Buy_EV'].max():.6f}]")
    print(f"  Pred mean: {sub['Will_Buy_EV'].mean():.6f}")
    print(f"  Pred std: {sub['Will_Buy_EV'].std():.6f}")

print("\nBoth submissions validated ✓")
