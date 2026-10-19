# v35 fold-checkpoint runner: run one fold per invocation so long jobs survive
# tool timeouts. Usage:
#   python3 train_v35_folds.py --stage pass1 --fold 0
#   python3 train_v35_folds.py --stage pass2 --fold 3
# State saved to scratch/v35_state/{stage}_f{fold}_{oof,test}.npy
import argparse, os, time
import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument("--stage", choices=["pass1", "pass2"], required=True)
ap.add_argument("--fold", type=int, required=True)
args = ap.parse_args()

T0 = time.time()
STATE = "scratch/v35_state"
os.makedirs(STATE, exist_ok=True)
TARGET, ID = "Will_Buy_EV", "id"
CATS = ["Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible",
        "Subsidy_Available", "Range_Anxiety_Level"]

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
y = (train[TARGET] == "Yes").astype(np.uint8).to_numpy()
N, Nt = len(y), len(test)

# ---------------- row-local features ----------------
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
print(f"data loaded ({time.time()-T0:.0f}s)", flush=True)

# ---------------- keys + global factorization ----------------
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

K_tr, K_te = build_keys(X_tr), build_keys(X_te)
TE_COLS = ["k_inc", "k_km", "k_km_int", "k_inc100", "k_inc1000",
           "k_inc_km", "k_inc100_city", "k_inc1000_anx", "k_inc100_sub",
           "k_km_int_home", "k_age_inc100"]
FREQ_COLS = ["k_inc", "k_km", "k_inc_km"]
CODES_tr, CODES_te, N_KEYS = {}, {}, {}
for c in TE_COLS + FREQ_COLS:
    combined = pd.concat([K_tr[c], K_te[c]], ignore_index=True)
    codes, uniq = pd.factorize(combined)
    CODES_tr[c] = codes[:N].astype(np.int32)
    CODES_te[c] = codes[N:].astype(np.int32)
    N_KEYS[c] = len(uniq)
print(f"keys factored ({time.time()-T0:.0f}s)", flush=True)

NUMERIC_BASE = [c for c in X_tr.columns if c not in CATS]
INCOME_ALL = train["Annual_Income_USD"].to_numpy(dtype=np.int64)
INCOME_TEST = test["Annual_Income_USD"].to_numpy(dtype=np.int64)

# ---------------- feature builders ----------------
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

def income_highres(fit_income, fit_y, valid_income, test_income, q=16384, smooth=10.0, seed=17):
    prior = float(fit_y.mean())
    edges = np.linspace(float(fit_income.min()), float(fit_income.max()), q + 1)
    fc = np.searchsorted(edges[1:-1], fit_income)
    vc = np.searchsorted(edges[1:-1], valid_income)
    tc = np.searchsorted(edges[1:-1], test_income)
    scale = max(len(edges) - 2, 1)
    fit_feats = np.zeros((len(fit_income), 8), np.float32)
    for ifit, iva in StratifiedKFold(5, shuffle=True, random_state=seed).split(fc, fit_y):
        stats = bin_statistics(fc[ifit], fit_y[ifit], len(edges), float(fit_y[ifit].mean()), smooth)
        fit_feats[iva, 0] = fc[iva] / scale
        fit_feats[iva, 1:] = stats[fc[iva]]
    stats = bin_statistics(fc, fit_y, len(edges), prior, smooth)
    v_feats = np.column_stack([vc / scale, stats[vc]]).astype(np.float32)
    t_feats = np.column_stack([tc / scale, stats[tc]]).astype(np.float32)
    return fit_feats, v_feats, t_feats

def te_block(fit_codes, fit_y, other_codes_list, n_keys, m_levels=(1.0, 10.0, 100.0), seed=17):
    W = len(m_levels)
    fit_out = np.zeros((len(fit_codes), W), np.float32)
    outs = [np.zeros((len(oc), W), np.float32) for oc in other_codes_list]
    prior_all = float(fit_y.mean())
    splits = list(StratifiedKFold(5, shuffle=True, random_state=seed).split(fit_codes, fit_y))
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

def window_rates(fit_income, fit_y, other_incomes, widths=(2, 5, 20, 50, 200), m=20.0, seed=17):
    prior_all = float(fit_y.mean())
    out_fit = np.zeros((len(fit_income), len(widths)), np.float32)
    for wi, w in enumerate(widths):
        for ifit, iva in StratifiedKFold(5, shuffle=True, random_state=seed).split(fit_income, fit_y):
            prior = float(fit_y[ifit].mean())
            inc2s = np.sort(fit_income[ifit])
            y2 = fit_y[ifit][np.argsort(fit_income[ifit])].astype(np.float64)
            cum2 = np.concatenate([[0.0], np.cumsum(y2)])
            cnt2 = np.arange(0, len(y2) + 1, dtype=np.float64)
            lo = np.searchsorted(inc2s, fit_income[iva] - w, side="left")
            hi = np.searchsorted(inc2s, fit_income[iva] + w, side="right")
            s = cum2[hi] - cum2[lo]; c = cnt2[hi] - cnt2[lo]
            out_fit[iva, wi] = ((s + m * prior) / (c + m)).astype(np.float32)
    outs = []
    o2 = np.argsort(fit_income)
    incs = fit_income[o2]; ys = fit_y[o2].astype(np.float64)
    cum = np.concatenate([[0.0], np.cumsum(ys)])
    cnt = np.arange(0, len(ys) + 1, dtype=np.float64)
    for oi in other_incomes:
        block = np.zeros((len(oi), len(widths)), np.float32)
        for wi, w in enumerate(widths):
            lo = np.searchsorted(incs, oi - w, side="left")
            hi = np.searchsorted(incs, oi + w, side="right")
            s = cum[hi] - cum[lo]; c = cnt[hi] - cnt[lo]
            block[:, wi] = ((s + m * prior_all) / (c + m)).astype(np.float32)
        outs.append(block)
    return out_fit, outs

def freq_block(fit_codes, other_codes_list, n_keys):
    logc = np.log1p(np.bincount(fit_codes, minlength=n_keys).astype(np.float32))
    fit_out = logc[fit_codes].reshape(-1, 1)
    outs = [logc[np.clip(oc, 0, None)].reshape(-1, 1) for oc in other_codes_list]
    return fit_out, outs

# ---------------- fold assignment ----------------
outer = StratifiedKFold(5, shuffle=True, random_state=42)
FOLD = np.zeros(N, dtype=np.int8)
for k, (_, va) in enumerate(outer.split(np.zeros(N), y)):
    FOLD[va] = k
fold = args.fold
fit_idx = np.where(FOLD != fold)[0]
valid_idx = np.where(FOLD == fold)[0]

if args.stage == "pass2":
    oof1 = np.load(f"{STATE}/pass1_oof.npy")
    split_labels = pd.qcut(oof1, q=10, labels=False, duplicates="drop").astype(np.int32)
else:
    split_labels = y

# split by split_labels (pass2) must use SAME fit/valid rows as seed-42 FOLD!
# We keep the frozen outer partition; re-stratification only matters for the
# inner splits of TE/statistics, so pass2 re-runs inner splits stratified by
# prediction deciles via a different inner seed. Implemented through env flag:
if args.stage == "pass2":
    INNER_SEED = 23
else:
    INNER_SEED = 17

yf = y[fit_idx]
tf = time.time()

# TE blocks
n_val, n_test = len(valid_idx), Nt
te_fit_m = np.zeros((len(fit_idx), len(TE_COLS) * 3), np.float32)
te_val_m = np.zeros((n_val, len(TE_COLS) * 3), np.float32)
te_test_m = np.zeros((n_test, len(TE_COLS) * 3), np.float32)
for ci, c in enumerate(TE_COLS):
    f_out, o_outs = te_block(CODES_tr[c][fit_idx], yf,
                             [CODES_tr[c][valid_idx], CODES_te[c]], N_KEYS[c], seed=INNER_SEED)
    te_fit_m[:, ci*3:(ci+1)*3] = f_out
    te_val_m[:, ci*3:(ci+1)*3] = o_outs[0]
    te_test_m[:, ci*3:(ci+1)*3] = o_outs[1]
print(f"TE done ({time.time()-tf:.0f}s)", flush=True)

# freq blocks
fr_cols = []
for c in FREQ_COLS:
    f_out, o_outs = freq_block(CODES_tr[c][fit_idx], [CODES_tr[c][valid_idx], CODES_te[c]], N_KEYS[c])
    fr_cols.append((f_out, o_outs))
fr_fit_m = np.hstack([f for f, _ in fr_cols])
fr_val_m = np.hstack([o[0] for _, o in fr_cols])
fr_test_m = np.hstack([o[1] for _, o in fr_cols])

# income high-res + window rates
inc_fit, inc_val = INCOME_ALL[fit_idx], INCOME_ALL[valid_idx]
hi_fit, hi_val, hi_test = income_highres(inc_fit, yf, inc_val, INCOME_TEST)
wr_fit, wr_others = window_rates(inc_fit, yf, [inc_val, INCOME_TEST], seed=INNER_SEED)
wr_val, wr_test = wr_others
print(f"stat features done ({time.time()-tf:.0f}s)", flush=True)

CAT_CODES_tr = {c: X_tr[c].astype("category").cat.codes.to_numpy().astype(np.int32) for c in CATS}
CAT_CODES_te = {c: X_te[c].astype("category").cat.codes.to_numpy().astype(np.int32) for c in CATS}
BASE_tr = {c: X_tr[c].to_numpy().astype(np.float32).reshape(-1, 1) for c in NUMERIC_BASE}
BASE_te = {c: X_te[c].to_numpy().astype(np.float32).reshape(-1, 1) for c in NUMERIC_BASE}

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
print(f"matrices {Xf.shape} built ({time.time()-tf:.0f}s)", flush=True)

LGB_PARAMS = dict(n_estimators=2500, learning_rate=0.02, max_depth=5, num_leaves=32,
                  min_child_samples=10, subsample=0.8, subsample_freq=1,
                  colsample_bytree=0.3, reg_alpha=0.071, reg_lambda=2.0,
                  max_bin=255, feature_pre_filter=False, random_state=42,
                  n_jobs=8, verbose=-1, metric="auc")

model = LGBMClassifier(**LGB_PARAMS)
model.fit(Xf, yf, eval_set=[(Xv, y[valid_idx])],
          callbacks=[lgb.early_stopping(100, first_metric_only=True, verbose=False)])
pv = model.predict_proba(Xv)[:, 1]
pt = model.predict_proba(Xt)[:, 1]
auc_f = roc_auc_score(y[valid_idx], pv)
print(f"[{args.stage} fold {fold}] AUC {auc_f:.6f} trees {model.best_iteration_} "
      f"({time.time()-tf:.0f}s)", flush=True)

np.save(f"{STATE}/{args.stage}_f{fold}_oof.npy", pv)
np.save(f"{STATE}/{args.stage}_f{fold}_test.npy", pt)
np.save(f"{STATE}/{args.stage}_f{fold}_auc.npy", np.array([auc_f]))
print(f"saved ({time.time()-T0:.0f}s total)", flush=True)
