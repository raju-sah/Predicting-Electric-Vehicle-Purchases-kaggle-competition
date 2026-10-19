"""v71-v72 FINAL v2: Maximum-diversity rank blend submissions.

Key insight from deep analysis:
- The JZ anchor + megayak band-swap gives 0.94656 on public
- Minor perturbations around v57 will just tie at 0.94656 or drop
- To BEAT 0.94656, we need either:
  A) A fundamentally better ranking (no public recipe achieves this)
  B) Different band-swap parameters that might get lucky
  
Practical strategy for 2 submissions:
1. v71: The EXACT megayak recipe reproduced from scratch using DefiAudit's
   rebuilt champion + megayak predictions. If DefiAudit's rebuilt champion
   is slightly different from JZ, the band-swap might land differently.
   
2. v72: A mega-diverse rank blend of ALL available predictions (11 models),
   then apply the four-band half-swap. The theory: if we can build a base
   that's 0.00001-0.00002 better raw, the band-swap amplifies the gain.
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
import hashlib

BASE = "/home/raju/AI-ML Projects/Kaggle Competition/Predicting Electric Vehicle Purchases"
TARGET = "Will_Buy_EV"
ID = "id"
N_TEST = 286571

test = pd.read_csv(f"{BASE}/test.csv")
sample = pd.read_csv(f"{BASE}/sample_submission.csv")

# ---- Load everything ----
jz = pd.read_csv(f"{BASE}/subs/v43_jz_base_exact.csv")[TARGET].to_numpy()
v57 = pd.read_csv(f"{BASE}/subs/v57_public_band_swap_exact.csv")[TARGET].to_numpy()
v68 = pd.read_csv(f"{BASE}/subs/v68_verified5_logit.csv")[TARGET].to_numpy()

sv = pd.read_csv(f"{BASE}/scratch/pull_sep16/ds_megayak6/test_six_views.csv")
rml = pd.read_csv(f"{BASE}/scratch/pull_sep16/ds_megayak6/test_realmlp_g.csv")
mega_A = sv["A_lgbm_triple_te_digits_3seed"].to_numpy()
mega_B = sv["B_xgb_on_A_features"].to_numpy()
mega_C = sv["C_no_digits_windows_lift_sm2_30_300"].to_numpy()
mega_D = sv["D_no_exact_key_ladder_windows"].to_numpy()
mega_E = sv["E_ladder25_250_2500_lift_sm5_50_500"].to_numpy()
mega_F = sv["F_exact_rate_as_init_score"].to_numpy()
mega_ens = sv["ensemble"].to_numpy()
realmlp_10f = rml["G_realmlp_10fold"].to_numpy()
realmlp_3s = rml["G_realmlp_3seed"].to_numpy()

nj_lgbm_v1 = pd.read_csv(f"{BASE}/scratch/pull_sep16/ds_najiama/Pure LGBM_V1_test.csv")[TARGET].to_numpy()
nj_lgbm_v3 = pd.read_csv(f"{BASE}/scratch/pull_sep16/ds_najiama/Pure LGBM_V3_test.csv")[TARGET].to_numpy()
nj_xgb_5f = pd.read_csv(f"{BASE}/scratch/pull_sep16/ds_najiama/XGBoost_Triple_TE_5folds_test.csv")[TARGET].to_numpy()
nj_xgb_10f = pd.read_csv(f"{BASE}/scratch/pull_sep16/ds_najiama/XGBoost_Triple_TE_10folds_test.csv")[TARGET].to_numpy()
nj_sergey = pd.read_csv(f"{BASE}/scratch/pull_sep16/ds_najiama/Sergey_LGBM_submission.csv")[TARGET].to_numpy()
nj_blend = pd.read_csv(f"{BASE}/scratch/pull_sep16/ds_najiama/01_submission.csv")[TARGET].to_numpy()

print("All predictions loaded.", flush=True)

# ---- Pre-rank everything ----
def rank01(v):
    return rankdata(v) / len(v)

ranks = {
    "jz": rank01(jz),
    "v68": rank01(v68),
    "mega_A": rank01(mega_A),
    "mega_B": rank01(mega_B),
    "mega_C": rank01(mega_C),
    "mega_D": rank01(mega_D),
    "mega_E": rank01(mega_E),
    "mega_F": rank01(mega_F),
    "mega_ens": rank01(mega_ens),
    "realmlp_10f": rank01(realmlp_10f),
    "realmlp_3s": rank01(realmlp_3s),
    "nj_lgbm_v1": rank01(nj_lgbm_v1),
    "nj_lgbm_v3": rank01(nj_lgbm_v3),
    "nj_xgb_5f": rank01(nj_xgb_5f),
    "nj_xgb_10f": rank01(nj_xgb_10f),
    "nj_sergey": rank01(nj_sergey),
    "nj_blend": rank01(nj_blend),
}

# The proven four-band half-swap params
BANDS = [(0.54, 0.04), (0.20, 0.06), (0.58, 0.04), (0.47, 0.06)]

def apply_band_swap(base_rank, ref_rank, bands=BANDS, swap_ratio=0.5):
    """Apply band-swap: in each band, blend base and ref ranks."""
    result = base_rank.copy()
    total = 0
    for center, hw in bands:
        lo, hi = center - hw, center + hw
        mask = (base_rank >= lo) & (base_rank <= hi)
        n = mask.sum()
        result[mask] = (1-swap_ratio) * base_rank[mask] + swap_ratio * ref_rank[mask]
        total += n
    return result, total

def rank_to_probs(rank_preds, ref_probs):
    """Map rank predictions back to probability space using ref distribution."""
    sorted_ref = np.sort(ref_probs)
    indices = (rankdata(rank_preds) - 1).astype(int).clip(0, len(ref_probs) - 1)
    return sorted_ref[indices]

# ---- SUBMISSION 1: Megayak-weights hill-climb ensemble + band-swap ----
print("\n=== SUBMISSION 1: Megayak-weights ensemble + band-swap ===")

# The megayak published recipe uses:
# Views A-F with specific weights, then the four-band half-swap vs JZ.
# But we have ADDITIONAL predictions (v68, najiama, realmlp).
# 
# Build the strongest possible base as a rank-space average of ALL models,
# weighted by estimated quality, then apply the standard band-swap.

# Weights inspired by megayak's published recipe + our diversity additions
# Megayak published weights (approximate): A: 0.25, B: 0.15, E: 0.15, F: 0.15, G: 0.15, rest: 0.15
# We add v68 (our verified ensemble) as a new diversity source
weights_v71 = {
    "jz": 0.25,       # The anchor (0.94651 raw)
    "mega_A": 0.15,    # Best single view
    "mega_B": 0.05,    # XGB on A features
    "mega_E": 0.10,    # Ladder features
    "mega_F": 0.05,    # Exact rate as init_score
    "v68": 0.15,       # Our verified ensemble (different error structure)
    "realmlp_10f": 0.10,  # Neural net diversity
    "nj_xgb_5f": 0.05,   # Independent XGB
    "nj_lgbm_v3": 0.05,  # Independent LGBM
    "nj_blend": 0.05,    # Independent blend
}

base_v71 = np.zeros(N_TEST)
for name, w in weights_v71.items():
    base_v71 += w * ranks[name]
print(f"Base v71 built from {len(weights_v71)} models")

# Apply band-swap using megayak ensemble as reference
ref_v71 = 0.5 * ranks["mega_ens"] + 0.3 * ranks["mega_A"] + 0.2 * ranks["realmlp_10f"]
swapped_v71, n_moved = apply_band_swap(rank01(base_v71), ref_v71)
print(f"Band-swap moved {n_moved} rows")

sub1_preds = rank_to_probs(swapped_v71, jz)

# Verify
rho_1, _ = spearmanr(sub1_preds, v57)
print(f"ρ(v71, v57) = {rho_1:.8f}")
print(f"ρ(v71, jz) = {spearmanr(sub1_preds, jz)[0]:.8f}")

sub1_df = pd.DataFrame({ID: test[ID], TARGET: sub1_preds})
assert sub1_df[ID].equals(sample[ID])
assert np.isfinite(sub1_df[TARGET]).all()
sub1_df.to_csv(f"{BASE}/subs/v71_megadiv_bandswap.csv", index=False)
print(f"Saved v71_megadiv_bandswap.csv")

# ---- SUBMISSION 2: JZ anchor + extended 6-band swap with wider coverage ----
print("\n=== SUBMISSION 2: JZ anchor + extended 6-band half-swap ===")

# The original 4 bands cover approximately: [0.14-0.26, 0.41-0.62]
# That leaves the extremes [0-0.14] and [0.62-1.0] untouched, 
# plus a gap at [0.26-0.41].
# 
# Hypothesis: extending to 2 additional bands in the gap region might help.
# We use the JZ anchor as base (not our blend, since JZ is proven 0.94651)
# and swap with a multi-model reference.

ref_v72 = 0.4 * ranks["mega_A"] + 0.3 * ranks["v68"] + 0.2 * ranks["realmlp_10f"] + 0.1 * ranks["nj_xgb_5f"]

# Extended bands: original 4 + 2 new ones in the gap
EXTENDED_BANDS = [
    (0.20, 0.06),   # Original: [0.14, 0.26]
    (0.33, 0.05),   # NEW: [0.28, 0.38] - gap region
    (0.47, 0.06),   # Original: [0.41, 0.53]
    (0.54, 0.04),   # Original: [0.50, 0.58]
    (0.58, 0.04),   # Original: [0.54, 0.62]
    (0.68, 0.05),   # NEW: [0.63, 0.73] - upper region
]

base_v72 = ranks["jz"].copy()
swapped_v72, n_moved = apply_band_swap(base_v72, ref_v72, bands=EXTENDED_BANDS, swap_ratio=0.5)
print(f"Extended band-swap moved {n_moved} rows")

sub2_preds = rank_to_probs(swapped_v72, jz)

rho_2, _ = spearmanr(sub2_preds, v57)
print(f"ρ(v72, v57) = {rho_2:.8f}")
print(f"ρ(v72, jz) = {spearmanr(sub2_preds, jz)[0]:.8f}")

sub2_df = pd.DataFrame({ID: test[ID], TARGET: sub2_preds})
assert sub2_df[ID].equals(sample[ID])
assert np.isfinite(sub2_df[TARGET]).all()
sub2_df.to_csv(f"{BASE}/subs/v72_extended_6band_swap.csv", index=False)
print(f"Saved v72_extended_6band_swap.csv")

# ---- Final comparison ----
print(f"\n=== FINAL COMPARISON ===")
rho_12, _ = spearmanr(sub1_preds, sub2_preds)
print(f"v71 vs v72: ρ = {rho_12:.8f}")
print(f"v71 vs v57: ρ = {spearmanr(sub1_preds, v57)[0]:.8f} (current best 0.94656)")
print(f"v72 vs v57: ρ = {spearmanr(sub2_preds, v57)[0]:.8f}")
print(f"v71 vs JZ:  ρ = {spearmanr(sub1_preds, jz)[0]:.8f} (0.94651 anchor)")
print(f"v72 vs JZ:  ρ = {spearmanr(sub2_preds, jz)[0]:.8f}")
print(f"v71 vs v68: ρ = {spearmanr(sub1_preds, v68)[0]:.8f} (OOF 0.94627)")
print(f"v72 vs v68: ρ = {spearmanr(sub2_preds, v68)[0]:.8f}")

# Row-level diff
for name, preds in [("v71", sub1_preds), ("v72", sub2_preds)]:
    d = np.abs(preds - v57)
    print(f"{name} vs v57: {(d > 0.001).sum()} rows differ >0.001, max={d.max():.6f}")
    d2 = np.abs(preds - jz)
    print(f"{name} vs JZ:  {(d2 > 0.001).sum()} rows differ >0.001, max={d2.max():.6f}")

# SHA
for name, path in [("v71", "v71_megadiv_bandswap.csv"), ("v72", "v72_extended_6band_swap.csv")]:
    sha = hashlib.sha256(open(f"{BASE}/subs/{path}", "rb").read()).hexdigest()
    print(f"{name}: sha256={sha[:32]}")

print("\n=== READY TO SUBMIT ===")
print(f"v71_megadiv_bandswap.csv: Multi-diversity base + proven 4-band swap")
print(f"v72_extended_6band_swap.csv: JZ anchor + extended 6-band swap (2 new bands)")
