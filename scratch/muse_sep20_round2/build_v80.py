"""v80: mega-champ 50/50 rank-blend reproduction (DefiAudit r8_m50 recipe).

w*rank01(mega) + (1-w)*rank01(champ), w=0.50.
mega = megayak six-views ensemble test; champ = DefiAudit r6_rebuilt_94651
(frozen 0.94651 champion). Author-measured 0.94657 public.
HONEST LABEL: public-probe reproduction. No train-fold predictions exist for
either input, so no complete comparable OOF is claimed. Submitted per explicit
user order; do not cite as OOF-validated.
"""
import hashlib

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

TARGET = "Will_Buy_EV"
sample = pd.read_csv("sample_submission.csv")
mega = pd.read_csv("scratch/pull_sep16/ds_megayak6/test_six_views.csv")
champ = pd.read_csv("scratch/muse_sep20_round2/defiaudit_realmlp/r6_rebuilt_94651.csv")
assert mega["id"].equals(sample["id"]) and champ["id"].equals(sample["id"])
rm = (rankdata(mega["ensemble"].to_numpy(float)) / len(mega))
rc = (rankdata(champ[TARGET].to_numpy(float)) / len(champ))
blend_rank = 0.5 * rm + 0.5 * rc
# map back to probability space via champion distribution (rank-preserving)
order = np.argsort(np.argsort(blend_rank))
preds = np.sort(champ[TARGET].to_numpy(float))[order]
frame = pd.DataFrame({"id": sample["id"], TARGET: preds})
assert frame.columns.tolist() == ["id", TARGET] and len(frame) == 286571
assert frame["id"].equals(sample["id"]) and frame["id"].is_unique
assert np.isfinite(preds).all() and ((preds >= 0) & (preds <= 1)).all()
frame.to_csv("subs/v80_mega_champ_50_50.csv", index=False)
v57 = pd.read_csv("subs/v57_public_band_swap_exact.csv")[TARGET].to_numpy()
print(f"v80 saved rho_vs_v57={spearmanr(preds, v57)[0]:.7f} "
      f"sha={hashlib.sha256(open('subs/v80_mega_champ_50_50.csv','rb').read()).hexdigest()[:16]}")
