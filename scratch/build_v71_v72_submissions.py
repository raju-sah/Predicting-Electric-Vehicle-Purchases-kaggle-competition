"""v71-v72: Two smart submissions.

Strategy based on Sep 20 research intelligence:
1. The honest OOF frontier is ~0.946426 (Den Pugovkin 15-view L1 stack)
2. Band-swaps risk private shakeup (Den Pugovkin, DefiAudit warnings)
3. Top 3 (#1 Thimsen 0.94675, #2 Deotte 0.94672) are private — no public recipe
4. Our verified v68 logit ensemble (OOF 0.946271) is the strongest honest local blend
5. The DefiAudit r8 rank-blend files give us access to a reconstructed champion anchor

Submission 1: Rank-blend of v68 logit ensemble with JZ anchor, with optimized band-swap
  → aims to beat 0.94656 public (test all 4 bands at slightly different parameters)
  
Submission 2: Conservative blend that maximizes private LB safety
  → rank-blend v68 ensemble + megayak m25 (which has milder band-swap),
    hedging against shakeup
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
import hashlib

BASE = "/home/raju/AI-ML Projects/Kaggle Competition/Predicting Electric Vehicle Purchases"
TARGET = "Will_Buy_EV"
ID = "id"

# Load all assets
test = pd.read_csv(f"{BASE}/test.csv")
sample = pd.read_csv(f"{BASE}/sample_submission.csv")
train = pd.read_csv(f"{BASE}/train.csv")
y = (train[TARGET] == "Yes").astype(np.int8).to_numpy()

# 1. Our verified v68 logit ensemble (OOF 0.946271, public 0.94645)
v68 = pd.read_csv(f"{BASE}/subs/v68_verified5_logit.csv")
v68_preds = v68[TARGET].to_numpy()
print(f"v68 loaded: {len(v68_preds)} rows")

# 2. JZ anchor (public 0.94651)
jz = pd.read_csv(f"{BASE}/subs/v43_jz_base_exact.csv")
jz_preds = jz[TARGET].to_numpy()
sha = hashlib.sha256(open(f"{BASE}/subs/v43_jz_base_exact.csv", "rb").read()).hexdigest()
print(f"JZ anchor loaded: {len(jz_preds)} rows, sha={sha[:16]}")

# 3. v57 band-swap (our current best, public 0.94656) 
v57 = pd.read_csv(f"{BASE}/subs/v57_public_band_swap_exact.csv")
v57_preds = v57[TARGET].to_numpy()
print(f"v57 loaded: {len(v57_preds)} rows")

# 4. DefiAudit rank blends (if available)
defi_m25 = pd.read_csv(f"{BASE}/scratch/defiaudit_realmlp/r8_out/r8_m25.csv")
defi_m10 = pd.read_csv(f"{BASE}/scratch/defiaudit_realmlp/r8_out/r8_m10.csv")
defi_mega = pd.read_csv(f"{BASE}/scratch/defiaudit_realmlp/r8_out/r8_m00_mega_verbatim.csv")
print(f"DefiAudit r8_m25 loaded: {len(defi_m25)} rows")

# 5. Megayak six-views test predictions
sv_test = pd.read_csv(f"{BASE}/scratch/pull_sep16/ds_megayak6/test_six_views.csv")
rml_test = pd.read_csv(f"{BASE}/scratch/pull_sep16/ds_megayak6/test_realmlp_g.csv")
mega_ensemble_test = sv_test["ensemble"].to_numpy()
mega_A_test = sv_test["A_lgbm_triple_te_digits_3seed"].to_numpy()
realmlp_10f_test = rml_test["G_realmlp_10fold"].to_numpy()
print(f"Megayak six-views + RealMLP test predictions loaded")

# 6. Najiama test predictions
nj_lgbm_v3_test = pd.read_csv(f"{BASE}/scratch/pull_sep16/ds_najiama/Pure LGBM_V3_test.csv")[TARGET].to_numpy()
nj_xgb5_test = pd.read_csv(f"{BASE}/scratch/pull_sep16/ds_najiama/XGBoost_Triple_TE_5folds_test.csv")[TARGET].to_numpy()
nj_blend_test = pd.read_csv(f"{BASE}/scratch/pull_sep16/ds_najiama/01_submission.csv")[TARGET].to_numpy()
print(f"Najiama test predictions loaded")

# ---- Analysis: understand the rank structure ----
print("\n=== Rank correlation analysis ===")
rank_v68 = rankdata(v68_preds) / len(v68_preds)
rank_jz = rankdata(jz_preds) / len(jz_preds)
rank_v57 = rankdata(v57_preds) / len(v57_preds)
rank_mega_ens = rankdata(mega_ensemble_test) / len(mega_ensemble_test)
rank_mega_A = rankdata(mega_A_test) / len(mega_A_test)
rank_realmlp = rankdata(realmlp_10f_test) / len(realmlp_10f_test)
rank_defi_m25 = rankdata(defi_m25[TARGET].to_numpy()) / len(defi_m25)

pairs = [
    ("v68", "JZ", rank_v68, rank_jz),
    ("v68", "v57", rank_v68, rank_v57),
    ("v68", "mega_ens", rank_v68, rank_mega_ens),
    ("v68", "mega_A", rank_v68, rank_mega_A),
    ("v68", "realmlp", rank_v68, rank_realmlp),
    ("JZ", "v57", rank_jz, rank_v57),
    ("JZ", "mega_A", rank_jz, rank_mega_A),
    ("v57", "defi_m25", rank_v57, rank_defi_m25),
]
for n1, n2, r1, r2 in pairs:
    rho, _ = spearmanr(r1, r2)
    print(f"  ρ({n1}, {n2}) = {rho:.8f}")

# ---- SUBMISSION 1: Enhanced band-swap with multiple anchors ----
print("\n=== Building Submission 1: Enhanced multi-anchor band-swap ===")

# The key insight: v57 succeeded because band-swap swaps the JZ anchor
# with a different model's ranking in specific rank bands where the
# alternative model has better ordering.
# 
# We can improve this by:
# 1. Using our v68 ensemble (which has OOF 0.946271 vs memory 0.946220)
#    as the SOURCE of the rank corrections instead of the raw megayak
# 2. Applying the same proven band parameters
# 3. Using probability-space blending in the bands for smoother corrections

# Start from the v57 (our current best public 0.94656) as anchor
# and inject v68 corrections in regions where v68 disagrees most with v57
anchor = v57_preds.copy()
rank_anchor = rankdata(anchor) / len(anchor)

# The proven four bands from megayak
bands = [
    (0.54, 0.04),  # [0.50, 0.58]
    (0.20, 0.06),  # [0.14, 0.26]
    (0.58, 0.04),  # [0.54, 0.62]
    (0.47, 0.06),  # [0.41, 0.53]
]

# Approach: Use rank-space blend of v57 + v68 within the bands
# v68 has stronger OOF → its rank corrections in uncertain regions may be better
v68_rank = rankdata(v68_preds) / len(v68_preds)

# Build the enhanced prediction
enhanced = rank_anchor.copy()
total_moved = 0
for center, hw in bands:
    lo, hi = center - hw, center + hw
    mask = (rank_anchor >= lo) & (rank_anchor <= hi)
    n_in = mask.sum()
    # Within these bands, blend v57 rank with v68 rank (70/30 split)
    enhanced[mask] = 0.70 * rank_anchor[mask] + 0.30 * v68_rank[mask]
    total_moved += n_in
    print(f"  Band [{lo:.2f},{hi:.2f}]: {n_in} rows, 70/30 blend v57+v68")

print(f"  Total rows adjusted: {total_moved}")

# Convert back to probability space by mapping enhanced ranks to the original
# prediction distribution
sort_order = np.argsort(anchor)
sorted_values = anchor[sort_order]
# Map the enhanced ranks back to the original probability scale
enhanced_probs = np.interp(enhanced, 
                           np.linspace(0, 1, len(anchor)),
                           np.sort(anchor))

sub1 = pd.DataFrame({ID: test[ID], TARGET: enhanced_probs})
assert sub1[ID].equals(sample[ID])
assert np.isfinite(sub1[TARGET]).all()
sub1.to_csv(f"{BASE}/subs/v71_enhanced_band_blend.csv", index=False)
print(f"\nSaved v71_enhanced_band_blend.csv")

# ---- SUBMISSION 2: Private-LB hedged multi-model rank blend ----
print("\n=== Building Submission 2: Private-LB hedged blend ===")

# Strategy: build a blend that would do well on BOTH public AND private test.
# Use rank-averaging across multiple diverse prediction sets with zero tie-breaks.
# No band-swap trick — this is the "honest" submission.

# Weight scheme based on estimated quality:
# - v68 (our strongest OOF 0.94627): primary anchor
# - mega_A (strongest single view): secondary anchor
# - JZ anchor (0.94651 public): strong reference
# - RealMLP (decorrelated NN): diversity member
# - Najiama blend (independent team): further diversity

components = {
    "v68": (v68_preds, 0.40),
    "mega_A": (mega_A_test, 0.20),
    "jz": (jz_preds, 0.15),
    "realmlp_10f": (realmlp_10f_test, 0.10),
    "nj_blend": (nj_blend_test, 0.08),
    "nj_xgb5": (nj_xgb5_test, 0.07),
}

# Rank-space blending (preserves AUC properties)
rank_blend = np.zeros(len(test))
for name, (preds, weight) in components.items():
    ranks = rankdata(preds) / len(preds)
    rank_blend += weight * ranks
    print(f"  {name:20s}: weight={weight:.2f}")

# Convert rank blend to probabilities using v68 as the calibration reference
# (v68 has the best calibration since it's from our verified OOF ensemble)
rank_of_blend = rankdata(rank_blend)
sorted_v68 = np.sort(v68_preds)
hedged_probs = sorted_v68[(rank_of_blend - 1).astype(int).clip(0, len(test) - 1)]

sub2 = pd.DataFrame({ID: test[ID], TARGET: hedged_probs})
assert sub2[ID].equals(sample[ID])
assert np.isfinite(sub2[TARGET]).all()
sub2.to_csv(f"{BASE}/subs/v72_hedged_rank_blend.csv", index=False)
print(f"\nSaved v72_hedged_rank_blend.csv")

# ---- Comparison ----
print("\n=== Final comparison ===")
rho_v71_v57, _ = spearmanr(sub1[TARGET].to_numpy(), v57_preds)
rho_v72_v57, _ = spearmanr(sub2[TARGET].to_numpy(), v57_preds)
rho_v71_v68, _ = spearmanr(sub1[TARGET].to_numpy(), v68_preds)
rho_v72_v68, _ = spearmanr(sub2[TARGET].to_numpy(), v68_preds)
rho_v71_jz, _ = spearmanr(sub1[TARGET].to_numpy(), jz_preds)
rho_v72_jz, _ = spearmanr(sub2[TARGET].to_numpy(), jz_preds)

print(f"v71 vs v57 (current best): ρ = {rho_v71_v57:.8f}")
print(f"v71 vs v68 (verified):     ρ = {rho_v71_v68:.8f}")
print(f"v71 vs JZ (anchor):        ρ = {rho_v71_jz:.8f}")
print(f"v72 vs v57 (current best): ρ = {rho_v72_v57:.8f}")
print(f"v72 vs v68 (verified):     ρ = {rho_v72_v68:.8f}")
print(f"v72 vs JZ (anchor):        ρ = {rho_v72_jz:.8f}")
print(f"v71 vs v72:                ρ = {spearmanr(sub1[TARGET].to_numpy(), sub2[TARGET].to_numpy())[0]:.8f}")

# How many rows differ significantly between v71/v72 and v57?
diff_v71 = np.abs(sub1[TARGET].to_numpy() - v57_preds)
diff_v72 = np.abs(sub2[TARGET].to_numpy() - v57_preds)
print(f"\nv71 vs v57: {(diff_v71 > 0.001).sum()} rows differ >0.001, max diff {diff_v71.max():.6f}")
print(f"v72 vs v57: {(diff_v72 > 0.001).sum()} rows differ >0.001, max diff {diff_v72.max():.6f}")

print("\n=== DONE ===")
print("v71_enhanced_band_blend.csv: Enhanced band-swap with v68 corrections (public push)")
print("v72_hedged_rank_blend.csv: Multi-model rank blend without band-swap (private hedge)")
