import pandas as pd
import numpy as np

v32 = pd.read_csv("subs/v32_simpson_grandmaster_apex.csv")["Will_Buy_EV"].values
v33 = pd.read_csv("subs/v33_probit_latent_apex.csv")["Will_Buy_EV"].values

diff = np.abs(v32 - v33)
print(f"Mean rank difference between linear rank blend (v32) and probit latent blend (v33): {diff.mean():.8f}")
print(f"Max rank difference: {diff.max():.8f}")
print(f"Number of rows with changed order: {(diff > 0).sum():,} / {len(v32):,}")
