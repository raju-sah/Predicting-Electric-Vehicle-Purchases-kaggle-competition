"""Round2: compare JZ five 0.94656 files vs v57 + each other (rank + value)."""
import hashlib
import io
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE = "scratch/muse_sep20_round2/jz_refresh"
z = zipfile.ZipFile(f"{BASE}/ranked_predictions_latest.zip.bin")
ids = ["56369364", "56369287", "56368879", "56368792", "56367897"]
frames = {}
for i, sid in enumerate(ids, 1):
    raw = z.read(f"own/own_00{i}_{sid}.csv")
    df = pd.read_csv(io.BytesIO(raw))
    frames[sid] = df
    print(sid, df.shape, list(df.columns),
          "shaOK:", hashlib.sha256(raw).hexdigest()[:12])
v57 = pd.read_csv("subs/v57_public_band_swap_exact.csv")
print("ids equal v57:", [f["id"].equals(v57["id"]) for f in frames.values()])
preds = {s: f["Will_Buy_EV"].to_numpy() for s, f in frames.items()}
v = v57["Will_Buy_EV"].to_numpy()
sids = list(preds)
print("rho vs v57:", {s: round(float(spearmanr(preds[s], v)[0]), 7) for s in sids})
print("exact-equal vs v57:", {s: bool(np.array_equal(preds[s], v)) for s in sids})
print("pairwise rho among JZ56:")
for i in range(5):
    print(" ", sids[i], [round(float(spearmanr(preds[sids[i]], preds[sids[j]])[0]), 7)
                         for j in range(5)])
