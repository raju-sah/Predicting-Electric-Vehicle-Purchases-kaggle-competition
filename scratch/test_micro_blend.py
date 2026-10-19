import pandas as pd
import numpy as np
from scipy.stats import rankdata

v11 = pd.read_csv('subs/v11_megayak_4view_daily.csv')['Will_Buy_EV'].values
nina = pd.read_csv('scratch/nina2025_out/submission.csv')['Will_Buy_EV'].values
deotte = pd.read_csv('scratch/deotte_gpu_out/sub_xgb_deotte_simpson.csv')['Will_Buy_EV'].values
realmlp = pd.read_csv('scratch/realmlp_out/submission.csv')['Will_Buy_EV'].values

r_v11 = (rankdata(v11) - 0.5) / len(v11)
r_nina = (rankdata(nina) - 0.5) / len(nina)
r_deotte = (rankdata(deotte) - 0.5) / len(deotte)
r_realmlp = (rankdata(realmlp) - 0.5) / len(realmlp)

# Test ultra-conservative micro-blend
blend = 0.88 * r_v11 + 0.06 * r_nina + 0.04 * r_deotte + 0.02 * r_realmlp
r_blend = (rankdata(blend) - 0.5) / len(blend)

print(f"Correlation with v11: {np.corrcoef(r_blend, r_v11)[0, 1]:.7f}")
print(f"Correlation with nina: {np.corrcoef(r_blend, r_nina)[0, 1]:.7f}")
print(f"Correlation with deotte: {np.corrcoef(r_blend, r_deotte)[0, 1]:.7f}")
print(f"Correlation with realmlp: {np.corrcoef(r_blend, r_realmlp)[0, 1]:.7f}")

diff = np.abs(rankdata(r_blend) - rankdata(r_v11))
print(f"Max rank change from v11: {diff.max()}, Mean rank change: {diff.mean():.2f}")
print(f"Fraction of rows with adjusted ranks: {(diff > 0).mean():.1%}")
