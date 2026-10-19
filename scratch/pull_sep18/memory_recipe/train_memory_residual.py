import hashlib, io, json, time, zipfile, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from lightgbm import LGBMClassifier
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.preprocessing import TargetEncoder

warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)

DATA_PATH = Path("/content/ev_s6e9")
REFERENCE_PATH = Path("/kaggle/input/s6e9-zoom-zoom-baseline")
TARGET, ID = "Will_Buy_EV", "id"

N_FOLDS = 5
INNER_FOLDS = 5
OUTER_SEED = 42
INNER_SEED = 17
MODEL_SEED = 42
THREADS = 6
VERBOSE_EVAL = 250
ANCHOR_MIN_COUNT = 10
PUBLIC_REFERENCE_WEIGHT = 0.80
BLEND_GRID = np.array([0.00, 0.25, 0.50, 0.75, 1.00])  # residual-model weight

CATEGORICAL_FEATURES = [
    "Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible",
    "Subsidy_Available", "Range_Anxiety_Level",
]

print("LightGBM", lgb.__version__)



def recipe_score(frame: pd.DataFrame) -> np.ndarray:
    anxiety = frame["Range_Anxiety_Level"].astype(str)
    subsidy = (frame["Subsidy_Available"].astype(str) == "Yes").astype(float)
    return (
        1.2 * frame["Annual_Income_USD"].to_numpy(float) / 100_000
        + 0.6 * frame["Environmental_Concern_Level"].to_numpy(float)
        + 2.0 * subsidy.to_numpy(float)
        - 1.0 * (anxiety == "Medium").to_numpy(float)
        - 3.0 * (anxiety == "High").to_numpy(float)
    )


def build_row_local_features(frame: pd.DataFrame):
    out = frame.copy().reset_index(drop=True)
    income = out["Annual_Income_USD"].to_numpy(np.int64)
    commute_x10 = np.rint(out["Daily_Commute_km"].to_numpy(float) * 10).astype(np.int64)

    # Token/digit views of synthetic numeric columns.
    for pos, divisor in enumerate((1, 10, 100, 1000, 10000, 100000), 1):
        out[f"inc_digit_{pos}"] = ((income // divisor) % 10).astype("int8")
    out["inc_mod100"] = (income % 100).astype("int16")
    out["inc_mod1000"] = (income % 1000).astype("int16")
    out["km_decimal_digit"] = (commute_x10 % 10).astype("int8")
    out["km_mod100"] = (commute_x10 % 100).astype("int16")
    for divisor in (50, 100, 250, 500, 1000, 2500, 5000):
        out[f"inc_q{divisor}"] = (income // divisor).astype("int32")
    for divisor in (5, 10, 25, 50):
        out[f"km_q{divisor}"] = (commute_x10 // divisor).astype("int32")

    out["is_30k_spike"] = (income == 30000).astype("int8")
    out["is_millionaire_cliff"] = (income >= 170537).astype("int8")
    out["is_dead_zone"] = ((income >= 38000) & (income <= 42000)).astype("int8")
    out["is_env_hater"] = (out["Environmental_Concern_Level"] == 1).astype("int8")
    out["recipe_score"] = recipe_score(out).astype("float32")
    out["total_public_chargers"] = (
        out["Charging_Stations_Near_Home"] + out["Charging_Stations_Near_Work"]
    ).astype("int16")
    out["home_station_access"] = (
        (out["Home_Charging_Possible"].astype(str) == "Yes").astype(np.int8)
        * (1 + out["Charging_Stations_Near_Home"])
    ).astype("int16")

    keys = pd.DataFrame(index=out.index)
    keys["k_inc_exact"] = income.astype(str)
    keys["k_inc100"] = (income // 100).astype(str)
    keys["k_inc1000"] = (income // 1000).astype(str)
    keys["k_km_x10"] = commute_x10.astype(str)
    keys["k_km_int"] = (commute_x10 // 10).astype(str)
    for col in CATEGORICAL_FEATURES + [
        "Age", "Number_of_Cars_Owned", "Charging_Stations_Near_Home",
        "Charging_Stations_Near_Work", "Environmental_Concern_Level",
    ]:
        keys[f"k_{col}"] = out[col].astype(str).to_numpy()
    return out, keys


def nearest_anchor(values, anchors):
    values = np.asarray(values, dtype=np.int64)
    anchors = np.asarray(anchors, dtype=np.int64)
    if len(anchors) == 0:
        raise ValueError("No income anchors recovered; lower ANCHOR_MIN_COUNT")
    j = np.clip(np.searchsorted(anchors, values), 1, len(anchors) - 1)
    left, right = anchors[j - 1], anchors[j]
    return np.where(np.abs(values - left) <= np.abs(right - values), left, right)


def map_zero(values, mapping):
    return values.map(mapping).fillna(0.0).astype(np.float32).to_numpy()


def as_string(series):
    return series.astype(str).reset_index(drop=True)



def alpha_one_target_encoding(train_keys, y_train, other_keys, columns):
    # Inner-fold encoding: a training row never contributes its own target.
    prior = float(np.mean(y_train))
    encoded_train = {c: np.full(len(train_keys), prior, np.float32) for c in columns}
    inner = KFold(INNER_FOLDS, shuffle=True, random_state=INNER_SEED)

    for inner_fit, inner_valid in inner.split(train_keys):
        for col in columns:
            stats = pd.DataFrame({"key": train_keys[col].iloc[inner_fit].to_numpy(),
                                  "y": y_train[inner_fit]}).groupby("key")["y"].agg(["sum", "size"])
            mapping = (stats["sum"] + prior) / (stats["size"] + 1.0)
            encoded_train[col][inner_valid] = (
                train_keys[col].iloc[inner_valid].map(mapping).fillna(prior).to_numpy(np.float32)
            )

    encoded_others = [dict() for _ in other_keys]
    for col in columns:
        stats = pd.DataFrame({"key": train_keys[col].to_numpy(), "y": y_train}).groupby("key")["y"].agg(["sum", "size"])
        mapping = (stats["sum"] + prior) / (stats["size"] + 1.0)
        for i, keys in enumerate(other_keys):
            encoded_others[i][col] = keys[col].map(mapping).fillna(prior).to_numpy(np.float32)
    return encoded_train, encoded_others


def bin_statistics(codes, y_fit, n_bins, prior, smooth):
    sums = np.bincount(codes, weights=y_fit, minlength=n_bins).astype(np.float64)
    counts = np.bincount(codes, minlength=n_bins).astype(np.float64)
    central = (sums + smooth * prior) / (counts + smooth)
    left_sums, left_counts = np.r_[0.0, sums[:-1]], np.r_[0.0, counts[:-1]]
    right_sums, right_counts = np.r_[sums[1:], 0.0], np.r_[counts[1:], 0.0]
    left = (left_sums + smooth * prior) / (left_counts + smooth)
    right = (right_sums + smooth * prior) / (right_counts + smooth)
    kernel = np.exp(-0.5 * (np.arange(-1, 2) / 0.8) ** 2)
    neighbor_sums = np.convolve(sums, kernel, mode="same")
    neighbor_counts = np.convolve(counts, kernel, mode="same")
    symmetric = (neighbor_sums + smooth * kernel.sum() * prior) / (neighbor_counts + smooth * kernel.sum())
    slope = right - left
    curvature = central - 0.5 * (left + right)
    return np.column_stack([central, symmetric, left, right, slope, curvature, np.log1p(counts)]).astype(np.float32)


def income_asymmetric_features(fit_income, fit_y, valid_income, test_income, q=16384, smooth=10.0):
    fit_income = np.asarray(fit_income, np.float64)
    valid_income = np.asarray(valid_income, np.float64)
    test_income = np.asarray(test_income, np.float64)
    fit_y = np.asarray(fit_y, np.uint8)
    prior = float(fit_y.mean())
    edges = np.linspace(float(fit_income.min()), float(fit_income.max()), q + 1)
    fit_codes = np.searchsorted(edges[1:-1], fit_income)
    valid_codes = np.searchsorted(edges[1:-1], valid_income)
    test_codes = np.searchsorted(edges[1:-1], test_income)
    n_bins = len(edges)
    scale = max(n_bins - 2, 1)
    fit_features = np.zeros((len(fit_income), 8), np.float32)
    inner = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=INNER_SEED)
    for inner_fit, inner_valid in inner.split(fit_codes, fit_y):
        stats = bin_statistics(fit_codes[inner_fit], fit_y[inner_fit], n_bins,
                               float(fit_y[inner_fit].mean()), smooth)
        fit_features[inner_valid, 0] = fit_codes[inner_valid] / scale
        fit_features[inner_valid, 1:] = stats[fit_codes[inner_valid]]
    stats = bin_statistics(fit_codes, fit_y, n_bins, prior, smooth)
    valid_features = np.column_stack([valid_codes / scale, stats[valid_codes]]).astype(np.float32)
    test_features = np.column_stack([test_codes / scale, stats[test_codes]]).astype(np.float32)
    return fit_features, valid_features, test_features



def prepare_features(fit_idx, valid_idx):
    train_part = X_rowlocal.iloc[fit_idx].reset_index(drop=True).copy()
    valid_part = X_rowlocal.iloc[valid_idx].reset_index(drop=True).copy()
    test_part = X_test_rowlocal.copy()
    train_keys = X_keys.iloc[fit_idx].reset_index(drop=True).copy()
    valid_keys = X_keys.iloc[valid_idx].reset_index(drop=True).copy()
    test_keys = X_test_keys.copy()
    y_fit = y[fit_idx]

    # Recover the source generator's income dictionary from frequent values in this fold only.
    fit_income = train_part["Annual_Income_USD"].astype(np.int64)
    counts = fit_income.value_counts()
    anchors = np.sort(counts[counts >= ANCHOR_MIN_COUNT].index.to_numpy(np.int64))
    for frame, keys in ((train_part, train_keys), (valid_part, valid_keys), (test_part, test_keys)):
        anchor = nearest_anchor(frame["Annual_Income_USD"].to_numpy(np.int64), anchors)
        frame["nearest_income_anchor"] = anchor.astype(np.int32)
        frame["income_anchor_distance"] = np.abs(frame["Annual_Income_USD"].to_numpy(np.int64) - anchor).astype(np.int16)
        keys["k_inc_anchor"] = anchor.astype(str)

    # Fold-local frequency encodings.
    for key in train_keys.columns:
        mapping = train_keys[key].value_counts(normalize=True, dropna=False)
        name = f"{key}_fe"
        train_part[name] = map_zero(train_keys[key], mapping)
        valid_part[name] = map_zero(valid_keys[key], mapping)
        test_part[name] = map_zero(test_keys[key], mapping)

    # Three proven smoothings for broad key coverage.
    for smooth, tag in (("auto", "auto"), (10.0, "10"), (100.0, "100")):
        enc = TargetEncoder(shuffle=True, cv=INNER_FOLDS, smooth=smooth, random_state=INNER_SEED)
        tr = enc.fit_transform(train_keys, y_fit)
        va = enc.transform(valid_keys)
        te = enc.transform(test_keys)
        for j, key in enumerate(train_keys.columns):
            train_part[f"{key}_te{tag}"] = tr[:, j].astype(np.float32)
            valid_part[f"{key}_te{tag}"] = va[:, j].astype(np.float32)
            test_part[f"{key}_te{tag}"] = te[:, j].astype(np.float32)

    # Low-prior encodings recover exact generator memory instead of washing it out.
    memory_keys = ["k_inc_exact", "k_inc100", "k_inc1000", "k_inc_anchor", "k_km_x10", "k_km_int"]
    tr_mem, other_mem = alpha_one_target_encoding(
        train_keys, y_fit, [valid_keys, test_keys], memory_keys
    )
    for key in memory_keys:
        train_part[f"{key}_te1"] = tr_mem[key]
        valid_part[f"{key}_te1"] = other_mem[0][key]
        test_part[f"{key}_te1"] = other_mem[1][key]

    # Preserve native categorical handling with identical category dictionaries.
    for col in CATEGORICAL_FEATURES:
        categories = sorted(as_string(train_part[col]).unique().tolist())
        dtype = pd.CategoricalDtype(categories=categories, ordered=False)
        train_part[col] = as_string(train_part[col]).astype(dtype)
        valid_part[col] = as_string(valid_part[col]).astype(dtype)
        test_part[col] = as_string(test_part[col]).astype(dtype)

    # High-resolution income-shape statistics from the strongest public model.
    income = train["Annual_Income_USD"].to_numpy()
    blocks = income_asymmetric_features(
        income[fit_idx], y_fit, income[valid_idx], test["Annual_Income_USD"].to_numpy(), q=16384
    )
    for frame, block in zip((train_part, valid_part, test_part), blocks):
        for j in range(1, block.shape[1]):
            frame[f"income_shape_{j}"] = block[:, j]

    # Public ablation found the units digit slightly harmful.
    return [f.drop(columns="inc_digit_1") for f in (train_part, valid_part, test_part)]



def make_model(seed):
    return LGBMClassifier(
        n_estimators=3500, learning_rate=0.02, max_depth=5, num_leaves=32,
        min_child_samples=10, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.3, reg_alpha=0.071, reg_lambda=2.0,
        max_bin=255, feature_pre_filter=False, random_state=seed,
        n_jobs=THREADS, verbose=-1, metric="auc",
    )


def fit_plain(x_fit, y_fit, x_valid, y_valid, fold):
    model = make_model(MODEL_SEED + fold)
    model.fit(
        x_fit, y_fit, eval_set=[(x_valid, y_valid)], eval_metric="auc",
        callbacks=[lgb.early_stopping(100, first_metric_only=True), lgb.log_evaluation(VERBOSE_EVAL)],
    )
    return model


def calibrated_recipe_logits(fit_frame, y_fit, *other_frames):
    fit_score = recipe_score(fit_frame).reshape(-1, 1)
    calibrator = LogisticRegression(C=1e4, solver="lbfgs", max_iter=1000)
    calibrator.fit(fit_score, y_fit)
    outputs = [calibrator.decision_function(fit_score)]
    outputs.extend(calibrator.decision_function(recipe_score(f).reshape(-1, 1)) for f in other_frames)
    return outputs


def fit_residual(x_fit, y_fit, x_valid, y_valid, fit_margin, valid_margin, fold):
    model = make_model(MODEL_SEED + 100 + fold)
    model.fit(
        x_fit, y_fit,
        init_score=fit_margin,
        eval_set=[(x_valid, y_valid)],
        eval_init_score=[valid_margin],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(100, first_metric_only=True), lgb.log_evaluation(VERBOSE_EVAL)],
    )
    return model


def residual_probability(model, features, margin):
    correction = model.predict(features, raw_score=True, num_iteration=model.best_iteration_)
    return expit(np.asarray(margin) + np.asarray(correction))



train = pd.read_csv(DATA_PATH / "train.csv")
test = pd.read_csv(DATA_PATH / "test.csv")
sample = pd.read_csv(DATA_PATH / "sample_submission.csv")

y = (train[TARGET].map({"No": 0, "Yes": 1})
     if not pd.api.types.is_numeric_dtype(train[TARGET]) else train[TARGET]).to_numpy(np.uint8)
X_source = train.drop(columns=[ID, TARGET]).reset_index(drop=True)
X_test_source = test.drop(columns=ID).reset_index(drop=True)
X_rowlocal, X_keys = build_row_local_features(X_source)
X_test_rowlocal, X_test_keys = build_row_local_features(X_test_source)

train_ids = train[ID].to_numpy()
test_ids = test[ID].to_numpy()

assert len(train) == 668665 and len(test) == 286571
assert sample.columns.tolist() == [ID, TARGET]
assert np.array_equal(sample[ID].to_numpy(), test_ids)
print("train", train.shape, "test", test.shape, "positive rate", y.mean())



splitter = StratifiedKFold(N_FOLDS, shuffle=True, random_state=OUTER_SEED)
oof_plain = np.zeros(len(train), np.float64)
oof_residual = np.zeros(len(train), np.float64)
test_plain_folds, test_residual_folds, fold_rows = [], [], []
fold_ids = np.full(len(train), -1, np.int8)
started = time.time()

for fold, (fit_idx, valid_idx) in enumerate(splitter.split(X_rowlocal, y)):
    print(f"\nFold {fold + 1}/{N_FOLDS}: preparing leak-free features", flush=True)
    x_fit, x_valid, x_test = prepare_features(fit_idx, valid_idx)
    print(f"{x_fit.shape[1]} features | {len(fit_idx):,} fit / {len(valid_idx):,} validation", flush=True)

    plain = fit_plain(x_fit, y[fit_idx], x_valid, y[valid_idx], fold)
    p_plain = plain.predict_proba(x_valid, num_iteration=plain.best_iteration_)[:, 1]
    t_plain = plain.predict_proba(x_test, num_iteration=plain.best_iteration_)[:, 1]

    fit_margin, valid_margin, test_margin = calibrated_recipe_logits(
        X_source.iloc[fit_idx], y[fit_idx], X_source.iloc[valid_idx], X_test_source
    )
    residual = fit_residual(
        x_fit, y[fit_idx], x_valid, y[valid_idx], fit_margin, valid_margin, fold
    )
    p_residual = residual_probability(residual, x_valid, valid_margin)
    t_residual = residual_probability(residual, x_test, test_margin)

    oof_plain[valid_idx] = p_plain
    oof_residual[valid_idx] = p_residual
    test_plain_folds.append(t_plain)
    test_residual_folds.append(t_residual)
    fold_ids[valid_idx] = fold
    fold_rows.append({
        "fold": fold,
        "plain_auc": roc_auc_score(y[valid_idx], p_plain),
        "residual_auc": roc_auc_score(y[valid_idx], p_residual),
        "plain_trees": int(plain.best_iteration_),
        "residual_trees": int(residual.best_iteration_),
    })
    print(pd.DataFrame(fold_rows).tail(1).to_string(index=False))
    print(f"elapsed {(time.time() - started) / 60:.1f} min", flush=True)
    del x_fit, x_valid, x_test, plain, residual

plain_auc = roc_auc_score(y, oof_plain)
residual_auc = roc_auc_score(y, oof_residual)
grid_rows = []
for w in BLEND_GRID:
    pred = (1 - w) * oof_plain + w * oof_residual
    grid_rows.append({"residual_weight": float(w), "oof_auc": roc_auc_score(y, pred)})
grid = pd.DataFrame(grid_rows)
best_weight = float(grid.loc[grid.oof_auc.idxmax(), "residual_weight"])
oof_final = (1 - best_weight) * oof_plain + best_weight * oof_residual

display(pd.DataFrame(fold_rows))
display(grid)
print(f"Plain OOF AUC:    {plain_auc:.9f}")
print(f"Residual OOF AUC: {residual_auc:.9f}")
print(f"Chosen weight:    {best_weight:.2f}")
print(f"Final OOF AUC:    {roc_auc_score(y, oof_final):.9f}")



test_plain = np.mean(test_plain_folds, axis=0)
test_residual = np.mean(test_residual_folds, axis=0)
test_model = (1 - best_weight) * test_plain + best_weight * test_residual

def validate_submission(frame, name):
    assert frame.columns.tolist() == [ID, TARGET], f"{name}: wrong columns"
    assert len(frame) == len(sample), f"{name}: wrong row count"
    assert frame[ID].is_unique, f"{name}: duplicate IDs"
    np.testing.assert_array_equal(frame[ID].to_numpy(), sample[ID].to_numpy())
    p = frame[TARGET].to_numpy(float)
    assert np.isfinite(p).all(), f"{name}: non-finite predictions"
    assert ((p >= 0) & (p <= 1)).all(), f"{name}: predictions outside [0, 1]"
    print(f"{name}: OK | {len(frame):,} rows | min={p.min():.6f} max={p.max():.6f}")

model_submission = pd.DataFrame({ID: test_ids, TARGET: test_model})
validate_submission(model_submission, "submission_model_only.csv")
model_submission.to_csv("submission_model_only.csv", index=False)

# Reproduce the public notebook's architecture with the improved OOF-gated model component.
catalog_file = REFERENCE_PATH / "ranked_catalog_latest.csv"
archive_file = REFERENCE_PATH / "ranked_predictions_latest.zip.bin"
if catalog_file.exists() and archive_file.exists():
    catalog = pd.read_csv(catalog_file)
    catalog["public_score"] = pd.to_numeric(catalog.public_score, errors="coerce")
    catalog = catalog[np.isfinite(catalog.public_score)].sort_values(
        ["public_score", "display_order"], ascending=[False, True], kind="stable"
    )
    best_record = catalog.iloc[0]
    with zipfile.ZipFile(archive_file) as archive:
        content = archive.read(best_record.csv_path)
    assert hashlib.sha256(content).hexdigest() == best_record.sha256
    reference = pd.read_csv(io.BytesIO(content)).set_index(ID, verify_integrity=True)
    ref_pred = reference.loc[test_ids, TARGET].to_numpy(float)
    final_pred = PUBLIC_REFERENCE_WEIGHT * ref_pred + (1 - PUBLIC_REFERENCE_WEIGHT) * test_model
    submission = pd.DataFrame({ID: test_ids, TARGET: final_pred})
    print(f"Reference: {best_record.author} submission {int(best_record.submission_id)} public {best_record.public_score:.5f}")
else:
    submission = model_submission.copy()
    print("Reference dataset not attached; submission.csv is the model-only prediction.")

validate_submission(submission, "submission.csv")
submission.to_csv("submission.csv", index=False)
pd.DataFrame({ID: train_ids, TARGET: y, "oof_plain": oof_plain,
              "oof_residual": oof_residual, "oof_final": oof_final,
              "fold": fold_ids}).to_csv("oof_predictions.csv", index=False)
pd.DataFrame(fold_rows).to_csv("fold_scores.csv", index=False)
grid.to_csv("blend_grid.csv", index=False)

metadata = {
    "plain_oof_auc": plain_auc,
    "residual_oof_auc": residual_auc,
    "final_oof_auc": roc_auc_score(y, oof_final),
    "residual_weight": best_weight,
    "folds": N_FOLDS,
    "public_reference_weight": PUBLIC_REFERENCE_WEIGHT if catalog_file.exists() else 0.0,
}
Path("run_metadata.json").write_text(json.dumps(metadata, indent=2))
display(submission.head())
print(json.dumps(metadata, indent=2))



