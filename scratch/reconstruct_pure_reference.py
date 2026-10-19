import pandas as pd
import numpy as np
import hashlib

print("--- RECONSTRUCTING EXACT PURE 0.94649 REFERENCE VECTOR ---")

sub = pd.read_csv("scratch/kernels_pulled/jazivxt_zoom_sep14/submission.csv")
pred = pd.read_csv("scratch/kernels_pulled/jazivxt_zoom_sep14/submission_lightgbm.csv")

y_sub = sub["Will_Buy_EV"].values
y_pred = pred["Will_Buy_EV"].values

# submission = 0.80 * shared + 0.20 * prediction
# shared = (submission - 0.20 * prediction) / 0.80
shared = (y_sub - 0.20 * y_pred) / 0.80

print(f"Reconstructed shared stats:")
print(f"  Min: {shared.min():.8f}, Max: {shared.max():.8f}, Mean: {shared.mean():.8f}")
print(f"  Unique: {len(np.unique(shared)):,}, Ties: {len(shared) - len(np.unique(shared)):,}")

df_shared = pd.DataFrame({"id": sub["id"], "Will_Buy_EV": shared})
df_shared.to_csv("scratch/reconstructed_shared_094649.csv", index=False)

with open("scratch/reconstructed_shared_094649.csv", "rb") as f:
    calc_hash = hashlib.sha256(f.read()).hexdigest()

print("Calculated hash:", calc_hash)
print("Target hash:     3a1ce448f12b6a6d42ff8dec7cc53a9b5e413b2327b8fe59f5097c688c513b85")
print(f"Hashes match: {calc_hash == '3a1ce448f12b6a6d42ff8dec7cc53a9b5e413b2327b8fe59f5097c688c513b85'}")
