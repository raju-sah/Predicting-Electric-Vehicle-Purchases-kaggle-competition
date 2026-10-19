"""v72: LightGBM DART + aggressive hyperparams + multi-seed.
DART (Dropouts meet Multiple Additive Regression Trees) provides genuine
decorrelation from regular GBDT by randomly dropping trees during training.
This produces predictions with different error structure.
Exact gate folds: StratifiedKFold(5, shuffle=True, random_state=42).
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import time

T0 = time.time()
TARGET = "Will_Buy_EV"
ID = "id"
CATS = ["Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible",
        "Subsidy_Available", "Range_Anxiety_Level"]
NUMERIC = ["Age", "Annual_Income_USD", "Daily_Commute_km", "Number_of_Cars_Owned",
           "Charging_Stations_Near_Home", "Charging_Stations_Near_Work",
           "Environmental_Concern_Level"]

train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
y = (train_df[TARGET] == "Yes").astype(np.int8).to_numpy()
N = len(y)
print(f"Data loaded: train {train_df.shape}, test {test_df.shape}", flush=True)

# ---- Feature Engineering (comprehensive) ----
def build_features(df):
    out = df.copy()
    income = df["Annual_Income_USD"].to_numpy(dtype=np.float64)
    commute = df["Daily_Commute_km"].to_numpy(dtype=np.float64)
    age = df["Age"].to_numpy(dtype=np.float64)
    cars = df["Number_of_Cars_Owned"].to_numpy(dtype=np.float64)
    ch = df["Charging_Stations_Near_Home"].to_numpy(dtype=np.float64)
    cw = df["Charging_Stations_Near_Work"].to_numpy(dtype=np.float64)
    env = df["Environmental_Concern_Level"].to_numpy(dtype=np.float64)
    
    income_int = np.rint(income).astype(np.int64)
    commute_x10 = np.round(commute * 10).astype(np.int64)
    
    # Digit features
    for k in range(-4, 4):
        power = 10 ** k
        out[f"inc_digit_{k}"] = np.mod(np.floor(income / power), 10).astype(np.int8)
    for k in range(-3, 3):
        power = 10 ** k
        out[f"km_digit_{k}"] = np.mod(np.floor(commute / power), 10).astype(np.int8)
    
    # Quantization
    out["inc_d2"] = (income_int // 10 % 10).astype("int8")
    out["inc_d3"] = (income_int // 100 % 10).astype("int8")
    out["inc_d4"] = (income_int // 1000 % 10).astype("int8")
    out["inc_mod100"] = (income_int % 100).astype("int16")
    out["inc_mod1000"] = (income_int % 1000).astype("int16")
    out["km_d1"] = (commute_x10 % 10).astype("int8")
    out["km_mod100"] = (commute_x10 % 100).astype("int8")
    out["age_mod5"] = (np.rint(age).astype(np.int64) % 5).astype("int8")
    
    for divisor in (50, 100, 250, 500, 1000, 2500, 5000):
        out[f"inc_q{divisor}"] = (income_int // divisor).astype("int32")
    for divisor in (5, 10, 25, 50):
        out[f"km_q{divisor}"] = (commute_x10 // divisor).astype("int32")
    
    # Boundary flags
    out["is_30k_spike"] = (income_int == 30000).astype("int8")
    out["is_millionaire_cliff"] = (income_int >= 170537).astype("int8")
    out["is_dead_zone"] = ((income_int >= 31004) & (income_int <= 41970)).astype("int8")
    out["is_env_hater"] = (env == 1).astype("int8")
    
    # Interaction features
    out["income_per_km"] = np.where(commute > 0, income / commute, 0).astype(np.float32)
    out["income_x_env"] = (income * env / 100000.0).astype(np.float32)
    out["charge_total"] = (ch + cw).astype(np.float32)
    out["charge_product"] = (ch * cw).astype(np.float32)
    out["charge_ratio_safe"] = np.where(cw > 0, ch / cw, ch).astype(np.float32)
    out["age_x_income"] = (age * income / 1e6).astype(np.float32)
    out["income_per_age"] = np.where(age > 0, income / age, 0).astype(np.float32)
    out["km_per_charge"] = np.where(ch + cw > 0, commute / (ch + cw), 0).astype(np.float32)
    out["env_x_charge"] = (env * (ch + cw)).astype(np.float32)
    out["cars_x_income"] = (cars * income / 100000.0).astype(np.float32)
    out["log_income"] = np.log1p(income).astype(np.float32)
    out["income_sq"] = ((income / 100000.0) ** 2).astype(np.float32)
    out["commute_sq"] = (commute ** 2).astype(np.float32)
    out["env_per_cars"] = np.where(cars > 0, env / cars, env).astype(np.float32)
    out["age_x_commute"] = (age * commute / 100.0).astype(np.float32)
    out["income_x_commute"] = (income * commute / 1e6).astype(np.float32)
    
    return out

X = build_features(train_df.drop(columns=[ID, TARGET]))
X_test = build_features(test_df.drop(columns=[ID]))

for c in CATS:
    X[c] = X[c].astype("category")
    X_test[c] = X_test[c].astype("category")

print(f"Features built: {X.shape[1]} ({time.time()-T0:.0f}s)", flush=True)

# ---- DART LightGBM Training ----
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
SEEDS = [42, 137, 2024]

# DART params - fundamentally different training dynamics
DART_PARAMS = dict(
    boosting_type='dart',
    n_estimators=2000,  # DART doesn't benefit from early stopping the same way
    learning_rate=0.03,
    num_leaves=95,
    min_child_samples=30,
    subsample=0.85,
    subsample_freq=1,
    colsample_bytree=0.7,
    reg_alpha=0.1,
    reg_lambda=2.0,
    max_bin=511,
    drop_rate=0.1,
    skip_drop=0.5,
    max_drop=50,
    verbose=-1,
    metric='auc',
    n_jobs=-1,
)

# Also train a regular GBDT with different hyperparams for diversity
GBDT_PARAMS = dict(
    boosting_type='gbdt',
    n_estimators=3500,
    learning_rate=0.015,
    num_leaves=127,
    max_depth=7,
    min_child_samples=15,
    subsample=0.75,
    subsample_freq=1,
    colsample_bytree=0.5,
    reg_alpha=0.2,
    reg_lambda=5.0,
    max_bin=511,
    verbose=-1,
    metric='auc',
    n_jobs=-1,
)

all_oofs = {}
all_tests = {}

for name, params_base in [("dart", DART_PARAMS), ("gbdt2", GBDT_PARAMS)]:
    oof_seeds = np.zeros((len(SEEDS), N), dtype=np.float64)
    test_seeds = np.zeros((len(SEEDS), len(test_df)), dtype=np.float64)
    
    for si, seed in enumerate(SEEDS):
        oof_s = np.zeros(N, dtype=np.float64)
        test_s = np.zeros(len(test_df), dtype=np.float64)
        params = params_base.copy()
        params['random_state'] = seed
        
        for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y)):
            tf = time.time()
            
            if name == "dart":
                # DART: use fixed iterations, no early stopping
                model = lgb.LGBMClassifier(**params)
                model.fit(X.iloc[tr_idx], y[tr_idx])
                best_iter = params['n_estimators']
            else:
                # GBDT: use early stopping
                model = lgb.LGBMClassifier(**params)
                model.fit(
                    X.iloc[tr_idx], y[tr_idx],
                    eval_set=[(X.iloc[va_idx], y[va_idx])],
                    eval_metric="auc",
                    callbacks=[lgb.early_stopping(150, verbose=False)],
                )
                best_iter = model.best_iteration_ or params['n_estimators']
            
            pv = model.predict_proba(X.iloc[va_idx])[:, 1]
            pt = model.predict_proba(X_test)[:, 1]
            oof_s[va_idx] = pv
            test_s += pt / 5
            
            auc_f = roc_auc_score(y[va_idx], pv)
            print(f"[{name} seed={seed}] fold {fold}: AUC {auc_f:.6f} trees {best_iter} ({time.time()-tf:.0f}s)", flush=True)
        
        oof_auc = roc_auc_score(y, oof_s)
        print(f"[{name} seed={seed}] OOF AUC: {oof_auc:.6f}", flush=True)
        oof_seeds[si] = oof_s
        test_seeds[si] = test_s
    
    oof_avg = oof_seeds.mean(axis=0)
    test_avg = test_seeds.mean(axis=0)
    oof_auc_avg = roc_auc_score(y, oof_avg)
    print(f"\n=== {name.upper()} MULTI-SEED OOF AUC: {oof_auc_avg:.7f} ===\n", flush=True)
    
    all_oofs[name] = oof_avg
    all_tests[name] = test_avg
    np.save(f"oof_v72_{name}_multiseed.npy", oof_avg)
    np.save(f"test_v72_{name}_multiseed.npy", test_avg)

# Blend DART + GBDT2
for w_dart in [0.3, 0.4, 0.5]:
    blend_oof = w_dart * all_oofs["dart"] + (1-w_dart) * all_oofs["gbdt2"]
    blend_auc = roc_auc_score(y, blend_oof)
    print(f"DART {w_dart:.1f} + GBDT2 {1-w_dart:.1f} OOF: {blend_auc:.7f}", flush=True)

print(f"\nTotal time: {time.time()-T0:.0f}s", flush=True)
