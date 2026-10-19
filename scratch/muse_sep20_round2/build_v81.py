"""v81: TRUE mega-champ 50/50 using megayak's own submission.csv (rank=v57).

DefiAudit's mega file rank-matches v57 (rho 1.0); our v80 wrongly used the
sixview ensemble column (rho 0.9979 vs v57). This is the faithful recipe.
Public-probe: NO complete OOF claimed. Submitted per user order.
"""
import hashlib

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

TARGET = "Will_Buy_EV"
sample = pd.read_csv("sample_submission.csv")
mg = pd.read_csv("scratch/pull_sep19/megayak56/out/submission.csv")
champ = pd.read_csv("scratch/muse_sep20_round2/defiaudit_realmlp/r6_rebuilt_94651.csv")
assert mg["id"].equals(sample["id"]) and champ["id"].equals(sample["id"])
b = 0.5 * rankdata(mg[TARGET].to_numpy(float)) / len(mg) + \
    0.5 * rankdata(champ[TARGET].to_numpy(float)) / len(champ)
order = np.argsort(np.argsort(b))
preds = np.sort(champ[TARGET].to_numpy(float))[order]
frame = pd.DataFrame({"id": sample["id"], TARGET: preds})
assert len(frame) == 286571 and frame["id"].equals(sample["id"])
assert np.isfinite(preds).all() and ((preds >= 0) & (preds <= 1)).all()
frame.to_csv("subs/v81_mega56_champ_50_50.csv", index=False)
v57 = pd.read_csv("subs/v57_public_band_swap_exact.csv")[TARGET].to_numpy()
print(f"v81 saved rho_vs_v57={spearmanr(preds, v57)[0]:.7f} "
      f"sha={hashlib.sha256(open('subs/v81_mega56_champ_50_50.csv','rb').read()).hexdigest()[:16]}")
