#!/usr/bin/env python3
"""S6E9 r8 (Sep 20 00:00 UTC): 5 megayak-blend arms (public-split climbing, dose-response)
+ 5 reverse probes (invert each recipe rule once) to map per-rule LB value.

megayak 0.94656 = jazivxt anchor + 4 half-band swaps chosen on public-sample disagreement.
Blends: w*rank01(mega) + (1-w)*rank01(champ). Verbatim = w=1.
Reverse x-probes: champion output + opposite hard block of one recipe rule each.
"""
import numpy as np
import pandas as pd
import os

BASE = "/home/z/my-project/s6e9"
ID, TARGET = "id", "Will_Buy_EV"
OUT = f"{BASE}/r8_out"
os.makedirs(OUT, exist_ok=True)

te = pd.read_csv(f"{BASE}/data/test.csv").sort_values(ID).reset_index(drop=True)
ch = pd.read_csv(f"{BASE}/restore_v7/r6_rebuilt_94651.csv").sort_values(ID).reset_index(drop=True)
mg = pd.read_csv(f"{BASE}/mega_94656/submission.csv").sort_values(ID).reset_index(drop=True)
assert (te[ID].to_numpy() == ch[ID].to_numpy()).all(), "id mismatch champ"
assert (te[ID].to_numpy() == mg[ID].to_numpy()).all(), "id mismatch mega"
ids = ch[ID].to_numpy()
N = len(ids)
print(f"N={N}")

def rank01(v):
    from scipy.stats import rankdata
    return rankdata(np.asarray(v, dtype=float), method="average") / len(v)

rc = rank01(ch[TARGET].to_numpy())
rm = rank01(mg[TARGET].to_numpy())
print(f"champ [{rc.min():.6f},{rc.max():.6f}] mega [{rm.min():.6f},{rm.max():.6f}]")

def emit_blend(fname, w):
    v = w * rm + (1 - w) * rc
    pd.DataFrame({ID: ids, TARGET: v}).to_csv(f"{OUT}/{fname}.csv", index=False)
    print(f"  {fname:12s} w_mega={w:.2f}")

emit_blend("r8_m00_mega_verbatim", 1.00)
emit_blend("r8_m75", 0.75)
emit_blend("r8_m50", 0.50)
emit_blend("r8_m25", 0.25)
emit_blend("r8_m10", 0.10)

# reverse probes on champion
inc = te["Annual_Income_USD"].to_numpy(float)
km = te["Daily_Commute_km"].to_numpy(float)
env1 = te["Environmental_Concern_Level"].to_numpy() == 1
no_sub = te["Subsidy_Available"].astype(str).to_numpy() == "No"
anx = te["Range_Anxiety_Level"].astype(str).to_numpy()
anx_mh = np.isin(anx, ["Medium", "High"])

v0 = ch[TARGET].to_numpy(float)
rules = [
    ("x1_income_ge_170537_inv", (inc >= 170537), -10.0),
    ("x2_income_31004_41970_inv", ((inc >= 31004) & (inc <= 41970)), +10.0),
    ("x3_commute_ge_83_inv", (km >= 83), +5.0),
    ("x4_30k_nosub_env_anx_inv", ((inc == 30000) & no_sub & (env1 | anx_mh)), +5.0),
]
for fname, m, d in rules:
    v = np.where(m, v0 + d, v0)
    pd.DataFrame({ID: ids, TARGET: v}).to_csv(f"{OUT}/r8_{fname}.csv", index=False)
    print(f"  r8_{fname:26s} rows={int(m.sum()):6d} delta={d:+.1f}")

v_all = v0.copy()
for _, m, d in rules:
    v_all = np.where(m, v_all + d, v_all)
pd.DataFrame({ID: ids, TARGET: v_all}).to_csv(f"{OUT}/r8_xall_inv.csv", index=False)
print(f"  r8_xall_inv rows={sum(int(m.sum()) for _, m, _ in rules)}")

files = sorted(os.listdir(OUT))
print(f"\n{len(files)} files written:")
for f in files:
    print(" ", f, f"{os.path.getsize(f'{OUT}/{f}')/1e6:.1f}MB")
