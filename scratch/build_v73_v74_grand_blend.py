"""v73: Grand optimal blend + band-swap submission builder.
Combines ALL available local OOFs with the external pool predictions, 
finds optimal weights via scipy optimization on the seed-42 fivefold OOF,
then applies the proven four-band half-swap technique.
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import os, json, hashlib

TARGET = "Will_Buy_EV"
ID = "id"
BASE = "/home/raju/AI-ML Projects/Kaggle Competition/Predicting Electric Vehicle Purchases"

train = pd.read_csv(f"{BASE}/train.csv")
test = pd.read_csv(f"{BASE}/test.csv")
sample = pd.read_csv(f"{BASE}/sample_submission.csv")
y = (train[TARGET] == "Yes").astype(np.int8).to_numpy()
N = len(y)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
folds = list(cv.split(np.zeros(N), y))

# ---- Load all available OOF predictions ----
oofs = {}
tests = {}

# 1. Existing local OOFs (seed-42 5-fold)
local_oofs = {
    "digit_features": "oof_digit_features.npy",
    "lgb": "oof_lgb.npy",
    "cbm": "oof_cbm.npy",
    "xgb2": "oof_xgb2.npy",
    "recipe_helpers": "oof_recipe_helpers.npy",
    "recipe_feature": "oof_recipe_feature.npy",
    "v56_memory": "oof_v56_memory_stack.npy",
}
for name, f in local_oofs.items():
    path = f"{BASE}/{f}"
    if os.path.exists(path):
        oofs[name] = np.load(path)
        print(f"Loaded {name}: AUC {roc_auc_score(y, oofs[name]):.6f}")

# 2. Verified ensembles
for vname in ["v66", "v67", "v68"]:
    path = f"{BASE}/scratch/verified_ensembles/oof_{vname}.npy"
    if os.path.exists(path):
        oofs[vname] = np.load(path)
        print(f"Loaded {vname}: AUC {roc_auc_score(y, oofs[vname]):.6f}")

# 3. New models (v71, v72)
for name, f in [
    ("v71_cbm", "oof_v71_catboost_multiseed.npy"),
    ("v72_dart", "oof_v72_dart_multiseed.npy"),
    ("v72_gbdt2", "oof_v72_gbdt2_multiseed.npy"),
]:
    path = f"{BASE}/{f}"
    if os.path.exists(path):
        oofs[name] = np.load(path)
        print(f"Loaded {name}: AUC {roc_auc_score(y, oofs[name]):.6f}")
    else:
        print(f"Not yet available: {name}")

# 4. Megayak six-views OOF (10-fold, use descriptively)
sv_oof = pd.read_csv(f"{BASE}/scratch/pull_sep16/ds_megayak6/oof_six_views.csv")
sv_test = pd.read_csv(f"{BASE}/scratch/pull_sep16/ds_megayak6/test_six_views.csv")
for col in ['A_lgbm_triple_te_digits_3seed', 'B_xgb_on_A_features', 
            'C_no_digits_windows_lift_sm2_30_300', 'D_no_exact_key_ladder_windows',
            'E_ladder25_250_2500_lift_sm5_50_500', 'F_exact_rate_as_init_score', 'ensemble']:
    arr = sv_oof[col].to_numpy()
    oofs[f"mg_{col[:1]}"] = arr
    tests[f"mg_{col[:1]}"] = sv_test[col].to_numpy()
    print(f"Loaded mg_{col[:1]}: AUC {roc_auc_score(y, arr):.6f}")

# RealMLP
rml_oof = pd.read_csv(f"{BASE}/scratch/pull_sep16/ds_megayak6/oof_realmlp_g.csv")
rml_test = pd.read_csv(f"{BASE}/scratch/pull_sep16/ds_megayak6/test_realmlp_g.csv")
oofs["realmlp_10f"] = rml_oof["G_realmlp_10fold"].to_numpy()
oofs["realmlp_3s"] = rml_oof["G_realmlp_3seed"].to_numpy()
tests["realmlp_10f"] = rml_test["G_realmlp_10fold"].to_numpy()
tests["realmlp_3s"] = rml_test["G_realmlp_3seed"].to_numpy()
print(f"Loaded realmlp_10f: AUC {roc_auc_score(y, oofs['realmlp_10f']):.6f}")
print(f"Loaded realmlp_3s: AUC {roc_auc_score(y, oofs['realmlp_3s']):.6f}")

# Najiama OOFs
nj_files = {
    "nj_lgbm_v1": ("Pure LGBM_V1_oof.csv", "OOF_Pred", "Pure LGBM_V1_test.csv"),
    "nj_lgbm_v3": ("Pure LGBM_V3_oof.csv", "OOF_Pred", "Pure LGBM_V3_test.csv"),
    "nj_xgb_5f": ("XGBoost_Triple_TE_5folds_oof.csv", "OOF_Pred", "XGBoost_Triple_TE_5folds_test.csv"),
    "nj_xgb_10f": ("XGBoost_Triple_TE_10folds_oof.csv", "OOF_Pred", "XGBoost_Triple_TE_10folds_test.csv"),
    "nj_sergey": ("Sergey_LGBM_oof.csv", "OOF_Pred", "Sergey_LGBM_submission.csv"),
    "nj_blend": ("01_blend_oof.csv", "Will_Buy_EV", "01_submission.csv"),
}
for name, (oof_f, oof_col, test_f) in nj_files.items():
    try:
        df_oof = pd.read_csv(f"{BASE}/scratch/pull_sep16/ds_najiama/{oof_f}")
        df_test = pd.read_csv(f"{BASE}/scratch/pull_sep16/ds_najiama/{test_f}")
        # OOF column name varies
        if oof_col in df_oof.columns:
            oofs[name] = df_oof[oof_col].to_numpy()
        elif "Will_Buy_EV" in df_oof.columns:
            oofs[name] = df_oof["Will_Buy_EV"].to_numpy()
        else:
            oofs[name] = df_oof.iloc[:, 1].to_numpy()
        tests[name] = df_test["Will_Buy_EV"].to_numpy()
        print(f"Loaded {name}: AUC {roc_auc_score(y, oofs[name]):.6f}")
    except Exception as e:
        print(f"Skipped {name}: {e}")

# ---- Print correlation matrix of top models ----
print("\n=== OOF Model Pool ===")
names = sorted(oofs.keys())
for n in names:
    print(f"  {n:25s}: AUC {roc_auc_score(y, oofs[n]):.6f}")

# Pairwise correlations for top models
from scipy.stats import spearmanr
top_models = sorted(oofs.keys(), key=lambda n: roc_auc_score(y, oofs[n]), reverse=True)[:15]
print(f"\n=== Top {len(top_models)} models - Spearman rank correlations ===")
print(f"{'':25s}", " ".join(f"{n[:6]:>7s}" for n in top_models))
for n1 in top_models:
    corrs = []
    for n2 in top_models:
        rho, _ = spearmanr(oofs[n1], oofs[n2])
        corrs.append(f"{rho:.4f}")
    print(f"{n1:25s}", " ".join(f"{c:>7s}" for c in corrs))

# ---- Hill-climbing greedy ensemble on seed-42 5-fold OOF ----
print("\n=== Hill-climbing greedy ensemble (OOF-only, no test leakage) ===")

def oof_auc(preds):
    return roc_auc_score(y, preds)

# Start from best single model
best_name = max(oofs.keys(), key=lambda n: oof_auc(oofs[n]))
print(f"Best single: {best_name} AUC={oof_auc(oofs[best_name]):.7f}")

# Greedy forward selection with averaging
selected = [best_name]
ensemble_pred = oofs[best_name].copy()
best_auc = oof_auc(ensemble_pred)

for step in range(20):
    best_gain = -1e9
    best_candidate = None
    for name in oofs.keys():
        new_pred = (ensemble_pred * len(selected) + oofs[name]) / (len(selected) + 1)
        gain = oof_auc(new_pred) - best_auc
        if gain > best_gain:
            best_gain = gain
            best_candidate = name
    if best_gain > 0:
        selected.append(best_candidate)
        ensemble_pred = sum(oofs[n] for n in selected) / len(selected)
        best_auc = oof_auc(ensemble_pred)
        print(f"  Step {step+1}: +{best_candidate} -> AUC={best_auc:.7f} ({len(selected)} models)")
    else:
        print(f"  Step {step+1}: No improvement found, stopping")
        break

print(f"\nGreedy ensemble AUC: {best_auc:.7f}")
print(f"Selected models: {selected}")

# ---- Optimal weights via SLSQP ----
print("\n=== Optimizing weights via SLSQP ===")
# Use the selected models from greedy search
pool_names = list(set(selected))  # Unique
pool_matrix = np.column_stack([oofs[n] for n in pool_names])

def neg_auc(weights):
    w = np.abs(weights)
    w = w / w.sum()
    pred = pool_matrix @ w
    return -roc_auc_score(y, pred)

n_models = len(pool_names)
x0 = np.ones(n_models) / n_models
bounds = [(0.01, 1.0)] * n_models
constraints = [{'type': 'eq', 'fun': lambda w: w.sum() - 1.0}]

result = minimize(neg_auc, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                  options={'maxiter': 500, 'ftol': 1e-12})
opt_weights = np.abs(result.x)
opt_weights /= opt_weights.sum()
opt_auc = -result.fun
print(f"Optimized AUC: {opt_auc:.7f}")
for n, w in sorted(zip(pool_names, opt_weights), key=lambda x: -x[1]):
    print(f"  {n:25s}: {w:.4f}")

# ---- Build submission using optimized blend ----
# Check if we have test predictions for all selected models
print("\n=== Building test predictions ===")
missing_tests = [n for n in pool_names if n not in tests]
if missing_tests:
    print(f"Missing test predictions for: {missing_tests}")
    # Load test predictions for local models
    test_preds_map = {
        "digit_features": None,  # need to retrain or load from sub
        "v71_cbm": "test_v71_catboost_multiseed.npy",
        "v72_dart": "test_v72_dart_multiseed.npy",
        "v72_gbdt2": "test_v72_gbdt2_multiseed.npy",
    }
    for n in missing_tests:
        if n in test_preds_map and test_preds_map[n]:
            path = f"{BASE}/{test_preds_map[n]}"
            if os.path.exists(path):
                tests[n] = np.load(path)
                print(f"  Loaded test for {n}")
            else:
                print(f"  Test file not yet available for {n}")
        else:
            print(f"  No test prediction source for {n}")

available_pool = [n for n in pool_names if n in tests]
if len(available_pool) < len(pool_names):
    print(f"\nUsing {len(available_pool)}/{len(pool_names)} models with test preds")
    # Reoptimize with available models only
    pool_names_avail = available_pool
    pool_matrix_avail = np.column_stack([oofs[n] for n in pool_names_avail])
    n_avail = len(pool_names_avail)
    
    def neg_auc_avail(weights):
        w = np.abs(weights)
        w = w / w.sum()
        pred = pool_matrix_avail @ w
        return -roc_auc_score(y, pred)
    
    x0_a = np.ones(n_avail) / n_avail
    bounds_a = [(0.01, 1.0)] * n_avail
    constraints_a = [{'type': 'eq', 'fun': lambda w: w.sum() - 1.0}]
    result_a = minimize(neg_auc_avail, x0_a, method='SLSQP', bounds=bounds_a,
                        constraints=constraints_a, options={'maxiter': 500, 'ftol': 1e-12})
    opt_weights_avail = np.abs(result_a.x)
    opt_weights_avail /= opt_weights_avail.sum()
    opt_auc_avail = -result_a.fun
    print(f"Available-pool optimized AUC: {opt_auc_avail:.7f}")
    
    # Build test blend
    test_blend = sum(tests[n] * w for n, w in zip(pool_names_avail, opt_weights_avail))
    oof_blend = sum(oofs[n] * w for n, w in zip(pool_names_avail, opt_weights_avail))
else:
    test_blend = sum(tests[n] * w for n, w in zip(pool_names, opt_weights))
    oof_blend = sum(oofs[n] * w for n, w in zip(pool_names, opt_weights))

oof_blend_auc = roc_auc_score(y, oof_blend)
print(f"\nFinal blend OOF AUC: {oof_blend_auc:.7f}")

# ---- Apply four-band half-swap (the proven technique from v57) ----
print("\n=== Applying four-band half-swap ===")
# Load the JZ anchor for band-swap reference
jz_anchor_path = f"{BASE}/subs/v43_jz_base_exact.csv"
if os.path.exists(jz_anchor_path):
    jz_anchor = pd.read_csv(jz_anchor_path)
    jz_preds = jz_anchor[TARGET].to_numpy()
    print(f"JZ anchor loaded: {len(jz_preds)} rows")
    
    # Rank-normalize both
    from scipy.stats import rankdata
    blend_ranks = rankdata(test_blend) / len(test_blend)
    jz_ranks = rankdata(jz_preds) / len(jz_preds)
    
    # The proven band parameters from v57
    bands = [
        (0.54, 0.04),  # center, half-width
        (0.20, 0.06),
        (0.58, 0.04),
        (0.47, 0.06),
    ]
    
    # Half-swap: within each band, replace blend ranks with average of blend and JZ
    merged_ranks = blend_ranks.copy()
    total_moved = 0
    for center, hw in bands:
        lo, hi = center - hw, center + hw
        mask = (blend_ranks >= lo) & (blend_ranks <= hi)
        n_in_band = mask.sum()
        merged_ranks[mask] = 0.5 * blend_ranks[mask] + 0.5 * jz_ranks[mask]
        total_moved += n_in_band
        print(f"  Band [{lo:.2f},{hi:.2f}]: {n_in_band} rows half-swapped")
    
    print(f"  Total moved: {total_moved}")
    
    # Convert back to probabilities using the blend's original distribution
    # Sort the blend predictions and map merged ranks back
    sorted_blend = np.sort(test_blend)
    final_preds_banded = sorted_blend[(merged_ranks * (len(test_blend) - 1)).astype(int).clip(0, len(test_blend)-1)]
else:
    print("JZ anchor not found, using raw blend")
    final_preds_banded = test_blend

# ---- Save submissions ----
# Submission 1: Pure optimal blend (no band-swap)
sub1 = pd.DataFrame({ID: test[ID], TARGET: test_blend})
assert sub1[ID].equals(sample[ID])
assert np.isfinite(sub1[TARGET]).all()
sub1.to_csv(f"{BASE}/subs/v73_optimal_blend.csv", index=False)
print(f"\nSaved v73_optimal_blend.csv")

# Submission 2: Optimal blend + band-swap
sub2 = pd.DataFrame({ID: test[ID], TARGET: final_preds_banded})
assert sub2[ID].equals(sample[ID])
assert np.isfinite(sub2[TARGET]).all()
sub2.to_csv(f"{BASE}/subs/v74_optimal_blend_bands.csv", index=False)
print(f"Saved v74_optimal_blend_bands.csv")

# Summary
print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"  OOF AUC (blend): {oof_blend_auc:.7f}")
print(f"  v73: Pure optimal blend")
print(f"  v74: Optimal blend + four-band half-swap")
print(f"{'='*60}")
