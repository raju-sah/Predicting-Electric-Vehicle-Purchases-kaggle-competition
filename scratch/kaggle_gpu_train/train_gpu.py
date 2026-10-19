import os, glob, time, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder
from sklearn.metrics import roc_auc_score
import xgboost as xgb

warnings.filterwarnings('ignore')

t0 = time.time()
def log(msg):
    print(f"[{time.time()-t0:7.1f}s] {msg}", flush=True)

# 1. Locate competition files
train_path = glob.glob('/kaggle/input/**/train.csv', recursive=True)[0]
test_path = glob.glob('/kaggle/input/**/test.csv', recursive=True)[0]
sample_path = glob.glob('/kaggle/input/**/sample_submission.csv', recursive=True)[0]

log("Loading raw data...")
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample = pd.read_csv(sample_path)

ID = "id"
TARGET = "Will_Buy_EV"
y = (train[TARGET] == "Yes").astype(np.int8).to_numpy()

NUMERIC = [
    "Age", "Annual_Income_USD", "Daily_Commute_km", "Number_of_Cars_Owned",
    "Charging_Stations_Near_Home", "Charging_Stations_Near_Work",
    "Environmental_Concern_Level",
]
CATEGORICAL = [
    "Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible",
    "Subsidy_Available", "Range_Anxiety_Level",
]

log(f"Train: {train.shape}, Test: {test.shape}")

# 2. Extract Exact Integer Digits (Fixing Topic 740824 float quotient error)
log("Decomposing exact integer digits...")
combined = pd.concat([train.drop(columns=[TARGET], errors='ignore'), test], ignore_index=True)

# Fixed non-negative integer digit decomposition:
for col in NUMERIC:
    v = combined[col].to_numpy(dtype=np.float64)
    scaled = np.rint(v * 10000.0).astype(np.int64)
    for k in range(-4, 4):
        combined[f"{col}_digit{k}"] = ((scaled // (10 ** (k + 4))) % 10).astype(np.int8)

# 3. Low-order arithmetic & quantization ladder
income = combined["Annual_Income_USD"].to_numpy(dtype=np.int64)
commute_x10 = np.round(combined["Daily_Commute_km"].to_numpy(dtype=float) * 10).astype(np.int64)

combined["inc_d1"] = (income % 10).astype(np.int8)
combined["inc_d2"] = (income // 10 % 10).astype(np.int8)
combined["inc_d3"] = (income // 100 % 10).astype(np.int8)
combined["inc_mod100"] = (income % 100).astype(np.int16)
combined["inc_mod1000"] = (income % 1000).astype(np.int16)

combined["km_d1"] = (commute_x10 % 10).astype(np.int8)
combined["km_mod100"] = (commute_x10 % 100).astype(np.int8)

for divisor in (50, 100, 250, 500, 1000, 2500, 5000):
    combined[f"inc_q{divisor}"] = (income // divisor).astype(np.int32)

for divisor in (5, 10, 25, 50):
    combined[f"km_q{divisor}"] = (commute_x10 // divisor).astype(np.int32)

# Spikes and cliffs
combined["is_30k_spike"] = (income == 30000).astype(np.int8)
combined["is_millionaire_cliff"] = (income >= 170537).astype(np.int8)
combined["is_dead_zone"] = ((income >= 38000) & (income <= 42000)).astype(np.int8)
combined["is_env_hater"] = (combined["Environmental_Concern_Level"] == 1).astype(np.int8)

# Frequency encoding on concatenated dataset
for c in CATEGORICAL:
    freq = combined[c].value_counts(normalize=True).to_dict()
    combined[f"{c}_freq"] = combined[c].map(freq).astype(np.float32)

# Smooth Keys for Target Encoding
te_keys = pd.DataFrame(index=combined.index)
te_keys["k_inc_exact"] = income.astype(str)
te_keys["k_inc100"] = (income // 100).astype(str)
te_keys["k_inc1000"] = (income // 1000).astype(str)
te_keys["k_km_int"] = (commute_x10 // 10).astype(str)

for col in CATEGORICAL + ["Age", "Number_of_Cars_Owned", "Charging_Stations_Near_Home", "Charging_Stations_Near_Work", "Environmental_Concern_Level"]:
    te_keys[f"k_{col}"] = combined[col].astype(str)

# Label encode original categorical features for XGBoost
for col in CATEGORICAL:
    mapping = {val: i for i, val in enumerate(combined[col].unique())}
    combined[col] = combined[col].map(mapping).astype(np.int8)

train_len = len(train)
X_all = combined.iloc[:train_len].drop(columns=[ID])
X_test_all = combined.iloc[train_len:].drop(columns=[ID])
te_keys_train = te_keys.iloc[:train_len]
te_keys_test = te_keys.iloc[train_len:]

log(f"Preprocessed features: {X_all.shape[1]}")

# 4. Stratified 10-Fold CV with Triple Target Encoding
FOLDS = 10
cv = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=42)
oof_preds = np.zeros(train_len, dtype=np.float64)
test_preds = np.zeros(len(test), dtype=np.float64)

xgb_params = dict(
    objective='binary:logistic',
    eval_metric='auc',
    tree_method='hist',
    device='cuda',
    learning_rate=0.015,
    max_depth=6,
    min_child_weight=15,
    subsample=0.85,
    colsample_bytree=0.35,
    reg_alpha=0.071,
    reg_lambda=2.0,
    max_bin=1024,
    n_estimators=4500,
    early_stopping_rounds=250,
    random_state=42,
)

log(f"Starting {FOLDS}-Fold GPU XGBoost training...")

for fold, (train_idx, val_idx) in enumerate(cv.split(X_all, y), 1):
    X_tr = X_all.iloc[train_idx].copy()
    X_va = X_all.iloc[val_idx].copy()
    X_te = X_test_all.copy()
    
    y_tr = y[train_idx]
    y_va = y[val_idx]
    
    # Triple Target Encoding inside fold
    for smooth, tag in [("auto", "auto"), (10.0, "10"), (100.0, "100")]:
        te = TargetEncoder(shuffle=True, cv=5, smooth=smooth, random_state=42)
        enc_tr = te.fit_transform(te_keys_train.iloc[train_idx], y_tr)
        enc_va = te.transform(te_keys_train.iloc[val_idx])
        enc_te = te.transform(te_keys_test)
        
        for idx, k in enumerate(te_keys.columns):
            X_tr[f"{k}_te{tag}"] = enc_tr[:, idx].astype(np.float32)
            X_va[f"{k}_te{tag}"] = enc_va[:, idx].astype(np.float32)
            X_te[f"{k}_te{tag}"] = enc_te[:, idx].astype(np.float32)
            
    model = xgb.XGBClassifier(**xgb_params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False
    )
    
    val_probs = model.predict_proba(X_va)[:, 1]
    oof_preds[val_idx] = val_probs
    test_preds += model.predict_proba(X_te)[:, 1] / FOLDS
    
    fold_auc = roc_auc_score(y_va, val_probs)
    log(f"Fold {fold:2d}/{FOLDS} - Trees: {model.best_iteration:4d} | AUC: {fold_auc:.6f}")

overall_auc = roc_auc_score(y, oof_preds)
log(f"=== OVERALL 10-FOLD OOF AUC: {overall_auc:.7f} ===")

# 5. Save artifacts
oof_df = pd.DataFrame({ID: train[ID], "oof_prediction": oof_preds, TARGET: y})
oof_df.to_csv("oof_xgb_fixed_digits.csv", index=False)

sub_df = pd.DataFrame({ID: test[ID], TARGET: test_preds})
sub_df.to_csv("submission.csv", index=False)
sub_df.to_csv("sub_xgb_fixed_digits.csv", index=False)

assert (sub_df[ID] == sample[ID]).all()
assert np.isfinite(sub_df[TARGET]).all()
assert ((sub_df[TARGET] >= 0) & (sub_df[TARGET] <= 1)).all()
log("All output files written and validated successfully!")
