"""Round2: locate JZ own 0.94656 files across archives; compare vs v57."""
import io
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE = "scratch/muse_sep20_round2/jz_refresh"
cands = ["ranked_predictions_latest.zip.bin", "ranked_scored_history.zip.bin"]
v57 = pd.read_csv("subs/v57_public_band_swap_exact.csv")["Will_Buy_EV"].to_numpy()
for archive in cands:
    try:
        z = zipfile.ZipFile(f"{BASE}/{archive}")
        names = z.namelist()
        hits = [n for n in names if "own_001_56369364" in n or "own_002_56369287" in n
                or "own_003_56368879" in n or "own_004" in n or "own_005" in n]
        print(archive, len(names), "hits:", hits[:6])
    except Exception as e:
        print(archive, "ERR", e)
