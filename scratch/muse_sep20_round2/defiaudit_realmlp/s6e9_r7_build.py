#!/usr/bin/env python3
"""S6E9 r7: rule-delta variants built on the surviving r6_rebuilt_94651.csv (0.94651).

Block semantics: base values v in (0,1) with ~1/N spacing. delta >= 1.0 = hard block
(jumps above/below ALL unshifted rows, matching the recipe's +/-5/+10 hard blocks);
|delta| < 1.0 = graded shift of ~delta*N rank positions.

Queue (10 files):
  r7_a1  Subsidy=No & income<100k            soft  -0.05   (train rate 0.1-1% vs g=17.5%)
  r7_a2  Subsidy=No & income<100k            hard  -1.00   (dose-response pair)
  r7_b1  commute 65..83 (join >=83 block)    hard  -1.00   (train dev -0.12..-0.14)
  r7_b2  commute 65..83                      soft  -0.05
  r7_c1  income 156k..170537 (join top blk)  hard  +1.00   (train rate ~0.49-0.54)
  r7_c2  income 140k..170537                 graded +0.30  (train rate ~0.33-0.39)
  r7_d1  existing 4 rules SOFT x0.0005             (hard vs graded test)
  r7_e1  Subsidy=Yes & income<50k            soft  -0.05   (train dev -0.105)
  r7_f1  combo: c1 top-ext + b1 commute-ext
  r7_g1  Range_Anxiety=High                  soft  -0.05   (train rate ~0.1%, n~2.2k)
"""
import numpy as np
import pandas as pd
import os

BASE = "/home/z/my-project/s6e9"
ID, TARGET = "id", "Will_Buy_EV"
OUT = f"{BASE}/r7_out"
os.makedirs(OUT, exist_ok=True)

te = pd.read_csv(f"{BASE}/data/test.csv").sort_values(ID).reset_index(drop=True)
r6 = pd.read_csv(f"{BASE}/r4_out/r6_rebuilt_94651.csv").sort_values(ID).reset_index(drop=True)
assert (te[ID].to_numpy() == r6[ID].to_numpy()).all(), "id mismatch"
v0 = r6[TARGET].to_numpy(float)
ids = r6[ID].to_numpy()
N = len(v0)
print(f"N={N} base [{v0.min():.6f},{v0.max():.6f}]")

inc = te["Annual_Income_USD"].to_numpy(float)
km = te["Daily_Commute_km"].to_numpy(float)
env1 = te["Environmental_Concern_Level"].to_numpy() == 1
no_sub = te["Subsidy_Available"].astype(str).to_numpy() == "No"
anx = te["Range_Anxiety_Level"].astype(str).to_numpy()
anx_mh = np.isin(anx, ["Medium", "High"])
anx_h = anx == "High"

# existing recipe rules (for r7_d1 soft-scaling)
base_rules = [
    ("income>=170537", (inc >= 170537), +10.0),
    ("income 31004..41970", ((inc >= 31004) & (inc <= 41970)), -10.0),
    ("commute>=83", (km >= 83), -5.0),
    ("30k&nosub&(env1|anxMH)", ((inc == 30000) & no_sub & (env1 | anx_mh)), -5.0),
]


def emit(fname, delta, desc):
    v = v0 + delta
    out = pd.DataFrame({ID: ids, TARGET: v})
    out.to_csv(f"{OUT}/{fname}.csv", index=False)
    moved = int((delta != 0).sum())
    print(f"  {fname:14s} rows={moved:7d} -> {desc}")


print("\nwriting r7 variants:")
emit("r7_a1_subno_soft", np.where(no_sub & (inc < 100000), -0.05, 0.0),
     "SubNo<100k -0.05 soft")
emit("r7_a2_subno_hard", np.where(no_sub & (inc < 100000), -1.00, 0.0),
     "SubNo<100k -1.0 hard block")
km_ext = (km >= 65) & (km < 83)
emit("r7_b1_commute_ext", np.where(km_ext, -1.00, 0.0),
     "commute65-83 join -5 block")
emit("r7_b2_commute_soft", np.where(km_ext, -0.05, 0.0), "commute65-83 -0.05 soft")
emit("r7_c1_topext", np.where((inc >= 156000) & (inc < 170537), +1.00, 0.0),
     "income156k-170k join +10 block")
emit("r7_c2_topmid", np.where((inc >= 140000) & (inc < 170537), +0.30, 0.0),
     "income140k-170k +0.30 graded")
# r7_d1: existing rules soft-scaled to graded deltas
delta_d = np.zeros(N)
for _, m, s in base_rules:
    delta_d += np.where(m, np.sign(s) * min(abs(s) * 0.0005, 0.005), 0.0)
emit("r7_d1_rules_soft", delta_d, "existing rules x0.0005 graded")
emit("r7_e1_ys_lowinc", np.where(~no_sub & (inc < 50000), -0.05, 0.0),
     "SubYes<50k -0.05 soft")
emit("r7_f1_combo", np.where((inc >= 156000) & (inc < 170537), +1.00, 0.0)
     + np.where(km_ext, -1.00, 0.0), "top-ext + commute-ext combo")
emit("r7_g1_anxhigh", np.where(anx_h, -0.05, 0.0), "RangeAnxietyHigh -0.05 soft")

# per-rule test counts for the log
print("\ntest-set counts:")
print(f"  SubNo&<100k : {(no_sub & (inc < 100000)).sum()}")
print(f"  commute65-83: {km_ext.sum()}")
print(f"  inc156-170.5k: {((inc >= 156000) & (inc < 170537)).sum()}")
print(f"  inc140-170.5k: {((inc >= 140000) & (inc < 170537)).sum()}")
print(f"  SubYes&<50k : {(~no_sub & (inc < 50000)).sum()}")
print(f"  anxHigh     : {anx_h.sum()}")
print("done ->", OUT)
