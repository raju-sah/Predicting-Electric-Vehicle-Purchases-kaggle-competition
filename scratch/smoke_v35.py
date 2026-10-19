# v35: "Zoom-Zoom-Max" — merged feature stack from all published winning recipes.
# Honest StratifiedKFold(5, seed=42) OOF; two-pass re-stratified training (jazivxt scheme).
# Families: digits/quantiles/modulo, boundary flags, cross-fitted triple-smoothed TEs on
# exact/ladder/composite keys, windowed target rates, 16384-bin income stats, frequencies.
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

T0 = time.time()
TARGET = "Will_Buy_EV"
ID = "id"
CATS = ["Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible",
        "Subsidy_Available", "Range_Anxiety_Level"]

train = pd.read_csv("train.csv").sample(frac=0.15, random_state=7).reset_index(drop=True)
test = pd.read_csv("test.csv").sample(frac=0.06, random_state=7).reset_index(drop=True)
y = (train[TARGET] == "Yes").astype(np.uint8).to_numpy()
N = len(y)
Nt = len(test)

outer = StratifiedKFold(5, shuffle=True, random_state=42)
FOLD = np.zeros(N, dtype=np.int8)
for k, (_, va) in enumerate(outer.split(np.zeros(N), y)):
    FOLD[va] = k

# ---------------- row-local numeric features (fit-free) ----------------
def build_rowlocal(df):
    out = df.copy()
    income = out["Annual_Income_USD"].to_numpy(dtype=np.int64)
    commute_x10 = np.round(out["Daily_Commute_km"].to_numpy(dtype=float) * 10).astype(np.int64)
    age = out["Age"].to_numpy(dtype=np.int64)
    out["inc_d2"] = (income // 10 % 10).astype("int8")
    out["inc_d3"] = (income // 100 % 10).astype("int8")
    out["inc_d4"] = (income // 1000 % 10).astype("int8")
    out["inc_mod100"] = (income % 100).astype("int16")
    out["inc_mod1000"] = (income % 1000).astype("int16")
    out["km_d1"] = (commute_x10 % 10).astype("int8")
    out["km_mod100"] = (commute_x10 % 100).astype("int8")
    out["age_mod5"] = (age % 5).astype("int8")
    for divisor in (50, 100, 250, 500, 1000, 2500, 5000):
        out[f"inc_q{divisor}"] = (income // divisor).astype("int32")
    for divisor in (5, 10, 25, 50):
        out[f"km_q{divisor}"] = (commute_x10 // divisor).astype("int32")
    out["is_30k_spike"] = (income == 30000).astype("int8")
    out["is_millionaire_cliff"] = (income >= 170537).astype("int8")
    out["is_dead_zone"] = ((income >= 31004) & (income <= 41970)).astype("int8")
    out["is_env_hater"] = (out["Environmental_Concern_Level"] == 1).astype("int8")
    out["km_x10"] = commute_x10.astype("int32")
    out["income_int"] = income.astype("int32")
    return out

X_tr = build_rowlocal(train.drop(columns=[ID, TARGET]))
X_te = build_rowlocal(test.drop(columns=[ID]))

# ---------------- keys: build strings once, factorize globally (label-free) ----------------
def build_keys(df):
    income = df["Annual_Income_USD"].to_numpy(dtype=np.int64)
    commute_x10 = np.round(df["Daily_Commute_km"].to_numpy(dtype=float) * 10).astype(np.int64)
    k = pd.DataFrame(index=df.index)
    k["k_inc"] = income.astype(str)
    k["k_km"] = commute_x10.astype(str)
    k["k_km_int"] = (commute_x10 // 10).astype(str)
    k["k_inc100"] = (income // 100).astype(str)
    k["k_inc1000"] = (income // 1000).astype(str)
    k["k_inc_km"] = k["k_inc"] + "_" + k["k_km_int"]
    k["k_inc100_city"] = k["k_inc100"] + "_" + df["City_Type"].astype(str)
    k["k_inc1000_anx"] = k["k_inc1000"] + "_" + df["Range_Anxiety_Level"].astype(str)
    k["k_inc100_sub"] = k["k_inc100"] + "_" + df["Subsidy_Available"].astype(str)
    k["k_km_int_home"] = k["k_km_int"] + "_" + df["Home_Charging_Possible"].astype(str)
    k["k_age_inc100"] = df["Age"].astype(str) + "_" + k["k_inc100"]
    return k

K_tr = build_keys(X_tr)
K_te = build_keys(X_te)

TE_COLS = ["k_inc", "k_km", "k_km_int", "k_inc100", "k_inc1000",
           "k_inc_km", "k_inc100_city", "k_inc1000_anx", "k_inc100_sub",
           "k_km_int_home", "k_age_inc100"]
FREQ_COLS = ["k_inc", "k_km", "k_inc_km"]

# global factorize over train+test so codes are shared across folds
CODES_tr, CODES_te, N_KEYS = {}, {}, {}
for c in TE_COLS + FREQ_COLS:
    combined = pd.concat([K_tr[c], K_te[c]], ignore_index=True)
    codes, uniq = pd.factorize(combined)
    CODES_tr[c] = codes[:N].astype(np.int32)
    CODES_te[c] = codes[N:].astype(np.int32)
    N_KEYS[c] = len(uniq)

NUMERIC_BASE = [c for c in X_tr.columns if c not in CATS]

# ---------------- [F5] cross-fitted high-res income bin statistics ----------------
def bin_statistics(codes, y_fit, n_bins, prior, smooth):
    sums = np.bincount(codes, weights=y_fit, minlength=n_bins).astype(np.float64)
    counts = np.bincount(codes, minlength=n_bins).astype(np.float64)
    central = (sums + smooth * prior) / (counts + smooth)
    left = (np.r_[0.0, sums[:-1]] + smooth * prior) / (np.r_[0.0, counts[:-1]] + smooth)
    right = (np.r_[sums[1:], 0.0] + smooth * prior) / (np.r_[counts[1:], 0.0] + smooth)
    kernel = np.exp(-0.5 * (np.arange(-1, 2) / 0.8) ** 2)
    nsum = np.convolve(sums, kernel, mode="same")
    ncnt = np.convolve(counts, kernel, mode="same")
    symmetric = (nsum + smooth * kernel.sum() * prior) / (ncnt + smooth * kernel.sum())
    slope = right - left
    curvature = central - 0.5 * (left + right)
    return np.column_stack([central, symmetric, left, right, slope, curvature,
                            np.log1p(counts)]).astype(np.float32)

INCOME_ALL = train["Annual_Income_USD"].to_numpy(dtype=np.int64)
INCOME_TEST = test["Annual_Income_USD"].to_numpy(dtype=np.int64)

def income_highres(fit_income, fit_y, valid_income, test_income, q=16384, smooth=10.0, seed=17):
    prior = float(fit_y.mean())
    edges = np.linspace(float(fit_income.min()), float(fit_income.max()), q + 1)
    fc = np.searchsorted(edges[1:-1], fit_income)
    vc = np.searchsorted(edges[1:-1], valid_income)
    tc = np.searchsorted(edges[1:-1], test_income)
    n_bins = len(edges)
    scale = max(n_bins - 2, 1)
    fit_feats = np.zeros((len(fit_income), 8), np.float32)
    inner = StratifiedKFold(5, shuffle=True, random_state=seed)
    for ifit, iva in inner.split(fc, fit_y):
        stats = bin_statistics(fc[ifit], fit_y[ifit], n_bins, float(fit_y[ifit].mean()), smooth)
        fit_feats[iva, 0] = fc[iva] / scale
        fit_feats[iva, 1:] = stats[fc[iva]]
    stats = bin_statistics(fc, fit_y, n_bins, prior, smooth)
    v_feats = np.column_stack([vc / scale, stats[vc]]).astype(np.float32)
    t_feats = np.column_stack([tc / scale, stats[tc]]).astype(np.float32)
    return fit_feats, v_feats, t_feats

# ---------------- [F3] triple-smoothed cross-fitted target encodings ----------------
def te_block(fit_codes, fit_y, other_codes_list, n_keys, m_levels=(1.0, 10.0, 100.0), seed=17):
    """Cross-fitted TEs on integer key codes. Fit side: inner-fold OOF.
    Other sides: encoded by full-fit stats; unseen keys -> prior."""
    W = len(m_levels)
    fit_out = np.zeros((len(fit_codes), W), np.float32)
    outs = [np.zeros((len(oc), W), np.float32) for oc in other_codes_list]
    prior_all = float(fit_y.mean())
    inner = StratifiedKFold(5, shuffle=True, random_state=seed)
    splits = list(inner.split(fit_codes, fit_y))
    for mi, m in enumerate(m_levels):
        for ifit, iva in splits:
            prior = float(fit_y[ifit].mean())
            sums = np.bincount(fit_codes[ifit], weights=fit_y[ifit], minlength=n_keys)
            counts = np.bincount(fit_codes[ifit], minlength=n_keys)
            te = (sums + m * prior) / (counts + m)
            fit_out[iva, mi] = te[fit_codes[iva]]
        sums = np.bincount(fit_codes, weights=fit_y, minlength=n_keys)
        counts = np.bincount(fit_codes, minlength=n_keys)
        te_full = (sums + m * prior_all) / (counts + m)
        for oi, oc in enumerate(other_codes_list):
            outs[oi][:, mi] = np.where(oc >= 0, te_full[np.clip(oc, 0, None)], prior_all)
    return fit_out, outs

# ---------------- [F4] windowed target rates around exact income ----------------
def window_rates(fit_income, fit_y, other_incomes, widths=(2, 5, 20, 50, 200), m=20.0, seed=17):
    """Rate of buyers among fit rows with income in [v-w, v+w].
    Fit side: inner-fold cross-fitted. Other sides: full-fit stats. All prior-smoothed."""
    prior_all = float(fit_y.mean())
    out_fit = np.zeros((len(fit_income), len(widths)), np.float32)
    inner = StratifiedKFold(5, shuffle=True, random_state=seed)
    splits = list(inner.split(fit_income, fit_y))
    for wi, w in enumerate(widths):
        for ifit, iva in splits:
            prior = float(fit_y[ifit].mean())
            inc2 = fit_income[ifit]
            o2 = np.argsort(inc2)
            inc2s = inc2[o2]
            y2 = fit_y[ifit][o2].astype(np.float64)
            cum2 = np.concatenate([[0.0], np.cumsum(y2)])
            cnt2 = np.arange(0, len(y2) + 1, dtype=np.float64)
            lo = np.searchsorted(inc2s, fit_income[iva] - w, side="left")
            hi = np.searchsorted(inc2s, fit_income[iva] + w, side="right")
            s = cum2[hi] - cum2[lo]
            c = cnt2[hi] - cnt2[lo]
            out_fit[iva, wi] = ((s + m * prior) / (c + m)).astype(np.float32)
    outs = []
    for oi in other_incomes:
        o2 = np.argsort(fit_income)
        incs = fit_income[o2]
        ys = fit_y[o2].astype(np.float64)
        cum = np.concatenate([[0.0], np.cumsum(ys)])
        cnt = np.arange(0, len(ys) + 1, dtype=np.float64)
        block = np.zeros((len(oi), len(widths)), np.float32)
        for wi, w in enumerate(widths):
            lo = np.searchsorted(incs, oi - w, side="left")
            hi = np.searchsorted(incs, oi + w, side="right")
            s = cum[hi] - cum[lo]
            c = cnt[hi] - cnt[lo]
            block[:, wi] = ((s + m * prior_all) / (c + m)).astype(np.float32)
        outs.append(block)
    return out_fit, outs

# ---------------- [F6] frequency encodings ----------------
def freq_block(fit_codes, other_codes_list, n_keys):
    counts = np.bincount(fit_codes, minlength=n_keys).astype(np.float32)
    logc = np.log1p(counts)
    fit_out = logc[fit_codes].reshape(-1, 1)
    outs = [logc[np.clip(oc, 0, None)].reshape(-1, 1) for oc in other_codes_list]
    return fit_out, outs

# ---------------- per-fold assembly + training ----------------
LGB_PARAMS = dict(n_estimators=2500, learning_rate=0.02, max_depth=5, num_leaves=32,
                  min_child_samples=10, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.3, reg_alpha=0.071, reg_lambda=2.0,
                  max_bin=255, feature_pre_filter=False, random_state=42,
                  n_jobs=8, verbose=-1, metric="auc")

def base_matrix(idx, from_test=False):
    src = X_te if from_test else X_tr
    part = src.iloc[idx] if idx is not None else src
    cols = []
    for c in NUMERIC_BASE:
        cols.append(part[c].to_numpy().astype(np.float32).reshape(-1, 1))
    for c in CATS:
        cols.append(part[c].astype("category").cat.codes.to_numpy().astype(np.int32).reshape(-1, 1))
    return np.hstack(cols)

CAT_CODES_tr = {c: X_tr[c].astype("category").cat.codes.to_numpy().astype(np.int32) for c in CATS}
CAT_CODES_te = {c: X_te[c].astype("category").cat.codes.to_numpy().astype(np.int32) for c in CATS}
BASE_tr = {c: X_tr[c].to_numpy().astype(np.float32).reshape(-1, 1) for c in NUMERIC_BASE}
BASE_te = {c: X_te[c].to_numpy().astype(np.float32).reshape(-1, 1) for c in NUMERIC_BASE}

def run_pass(split_labels, pass_name):
    oof = np.zeros(N, dtype=np.float64)
    test_fold_preds = []
    fold_aucs = []
    splitter = StratifiedKFold(5, shuffle=True, random_state=42)
    for fold, (fit_idx, valid_idx) in enumerate(splitter.split(X_tr, split_labels)):
        tf = time.time()
        yf = y[fit_idx]
        # target-side blocks (TE per key column; freq per key column)
        n_val, n_test = len(valid_idx), Nt
        te_fit_m = np.zeros((len(fit_idx), len(TE_COLS) * 3), np.float32)
        te_val_m = np.zeros((n_val, len(TE_COLS) * 3), np.float32)
        te_test_m = np.zeros((n_test, len(TE_COLS) * 3), np.float32)
        for ci, c in enumerate(TE_COLS):
            f_out, o_outs = te_block(CODES_tr[c][fit_idx], yf,
                                     [CODES_tr[c][valid_idx], CODES_te[c]], N_KEYS[c])
            te_fit_m[:, ci * 3:(ci + 1) * 3] = f_out
            te_val_m[:, ci * 3:(ci + 1) * 3] = o_outs[0]
            te_test_m[:, ci * 3:(ci + 1) * 3] = o_outs[1]
        fr_fit_m, fr_others = freq_block(CODES_tr["k_inc"][fit_idx],
                                         [CODES_tr["k_inc"][valid_idx], CODES_te["k_inc"]],
                                         N_KEYS["k_inc"])
        fr2_fit, fr2_others = freq_block(CODES_tr["k_km"][fit_idx],
                                         [CODES_tr["k_km"][valid_idx], CODES_te["k_km"]],
                                         N_KEYS["k_km"])
        fr3_fit, fr3_others = freq_block(CODES_tr["k_inc_km"][fit_idx],
                                         [CODES_tr["k_inc_km"][valid_idx], CODES_te["k_inc_km"]],
                                         N_KEYS["k_inc_km"])
        fr_fit_m = np.hstack([fr_fit_m, fr2_fit, fr3_fit])
        fr_val_m = np.hstack([fr_others[0], fr2_others[0], fr3_others[0]])
        fr_test_m = np.hstack([fr_others[1], fr2_others[1], fr3_others[1]])
        inc_fit = INCOME_ALL[fit_idx]
        inc_val = INCOME_ALL[valid_idx]
        hi_fit, hi_val, hi_test = income_highres(inc_fit, yf, inc_val, INCOME_TEST)
        wr_fit, wr_others = window_rates(inc_fit, yf, [inc_val, INCOME_TEST])
        wr_val, wr_test = wr_others

        def assemble(idx, from_test, te_m, fr_m, wr_m, hi_m):
            cols = []
            if from_test:
                cols += [BASE_te[c] for c in NUMERIC_BASE]
                cols += [CAT_CODES_te[c].reshape(-1, 1) for c in CATS]
            else:
                cols += [BASE_tr[c][idx] for c in NUMERIC_BASE]
                cols += [CAT_CODES_tr[c][idx].reshape(-1, 1) for c in CATS]
            cols += [te_m, fr_m, wr_m, hi_m]
            return np.hstack(cols)

        Xf = assemble(fit_idx, False, te_fit_m, fr_fit_m, wr_fit, hi_fit)
        Xv = assemble(valid_idx, False, te_val_m, fr_val_m, wr_val, hi_val)
        Xt = assemble(None, True, te_test_m, fr_test_m, wr_test, hi_test)

        model = LGBMClassifier(**LGB_PARAMS)
        model.fit(Xf, yf, eval_set=[(Xv, y[valid_idx])],
                  callbacks=[lgb.early_stopping(100, first_metric_only=True, verbose=False)])
        pv = model.predict_proba(Xv)[:, 1]
        oof[valid_idx] = pv
        test_fold_preds.append(model.predict_proba(Xt)[:, 1])
        auc_f = roc_auc_score(y[valid_idx], pv)
        fold_aucs.append(auc_f)
        print(f"[{pass_name}] fold {fold}: AUC {auc_f:.6f} trees {model.best_iteration_} "
              f"feats {Xf.shape[1]} ({time.time()-tf:.0f}s)", flush=True)
        del Xf, Xv, Xt, model, te_fit_m, te_val_m, te_test_m
    print(f"[{pass_name}] OOF AUC: {roc_auc_score(y, oof):.6f}", flush=True)
    return oof, np.mean(test_fold_preds, axis=0), fold_aucs

print("=== PASS 1 (plain stratified) ===", flush=True)
oof1, test1, aucs1 = run_pass(y, "pass1")
print(f"pass1 folds: {[f'{a:.5f}' for a in aucs1]}", flush=True)

split2 = pd.qcut(oof1, q=10, labels=False, duplicates="drop").astype(np.int32)
print("=== PASS 2 (re-stratified on pass-1 deciles) ===", flush=True)
oof2, test2, aucs2 = run_pass(split2, "pass2")

pass  # smoke: no save
pass
pass
pass
print(f"FINAL v35 pass2 OOF {roc_auc_score(y, oof2):.6f} | pass1 {roc_auc_score(y, oof1):.6f}")
print(f"total {time.time()-T0:.0f}s")
