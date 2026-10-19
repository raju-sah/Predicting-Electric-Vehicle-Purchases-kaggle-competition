import pandas as pd
import numpy as np
from scipy.stats import rankdata

v11 = pd.read_csv('subs/v11_megayak_4view_daily.csv')['Will_Buy_EV'].values
nina = pd.read_csv('scratch/nina2025_out/submission.csv')['Will_Buy_EV'].values

print(f"v11 min={v11.min():.5f}, max={v11.max():.5f}, mean={v11.mean():.5f}")
print(f"nina min={nina.min():.5f}, max={nina.max():.5f}, mean={nina.mean():.5f}")
corr = np.corrcoef(rankdata(v11), rankdata(nina))[0, 1]
print(f"Spearman rank correlation v11 vs nina: {corr:.7f}")

# Check difference in ranks
r_v11 = rankdata(v11)
r_nina = rankdata(nina)
diff = np.abs(r_v11 - r_nina)
print(f"Max rank change: {diff.max()}, Mean rank change: {diff.mean():.2f}")
print(f"Number of rows with changed ranks: {(diff > 0).sum()} / {len(v11)} ({(diff > 0).mean():.1%})")
