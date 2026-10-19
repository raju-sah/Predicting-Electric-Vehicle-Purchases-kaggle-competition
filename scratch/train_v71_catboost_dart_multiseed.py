"""v71: CatBoost Ordered Boosting + novel interactions + multi-seed.
Goal: produce a genuinely decorrelated, strong single model for blending.
Key differences from existing pool:
1. CatBoost ordered boosting (fundamentally different from LGB/XGB)
2. Novel ratio/product interactions not yet tried
3. Multi-seed averaging (3 seeds) for variance reduction
4. Deeper trees with more aggressive regularization
Exact gate folds: StratifiedKFold(5, shuffle=True, random_state=42).
"""
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
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

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
y = (train[TARGET] == "Yes").astype(np.int8).to_numpy()
N = len(y)
print(f"Data loaded: train {train.shape}, test {test.shape} ({time.time()-T0:.0f}s)", flush=True)

# ---- Feature Engineering ----
def build_features(df):
    out = df.copy()
    income = df["Annual_Income_USD"].to_numpy(dtype=np.float64)
    commute = df["Daily_Commute_km"].to_numpy(dtype=np.float64)
    age = df["Age"].to_numpy(dtype=np.float64)
    cars = df["Number_of_Cars_Owned"].to_numpy(dtype=np.float64)
    charge_home = df["Charging_Stations_Near_Home"].to_numpy(dtype=np.float64)
    charge_work = df["Charging_Stations_Near_Work"].to_numpy(dtype=np.float64)
    env = df["Environmental_Concern_Level"].to_numpy(dtype=np.float64)
    
    # Digit features (proven most valuable)
    income_int = np.rint(income).astype(np.int64)
    commute_x10 = np.round(commute * 10).astype(np.int64)
    
    for k in range(-4, 4):
        out[f"inc_digit_{k}"] = ((income_int // (10**max(0,k))) % 10 if k >= 0 
                                 else (np.rint(income * 10**(-k)).astype(np.int64) % 10)).astype(np.int8)
    
    out["inc_d2"] = (income_int // 10 % 10).astype("int8")
    out["inc_d3"] = (income_int // 100 % 10).astype("int8")
    out["inc_d4"] = (income_int // 1000 % 10).astype("int8")
    out["inc_mod100"] = (income_int % 100).astype("int16")
    out["inc_mod1000"] = (income_int % 1000).astype("int16")
    out["km_d1"] = (commute_x10 % 10).astype("int8")
    out["km_mod100"] = (commute_x10 % 100).astype("int8")
    out["age_mod5"] = (np.rint(age).astype(np.int64) % 5).astype("int8")
    
    # Quantization ladder
    for divisor in (50, 100, 250, 500, 1000, 2500, 5000):
        out[f"inc_q{divisor}"] = (income_int // divisor).astype("int32")
    for divisor in (5, 10, 25, 50):
        out[f"km_q{divisor}"] = (commute_x10 // divisor).astype("int32")
    
    # Boundary flags
    out["is_30k_spike"] = (income_int == 30000).astype("int8")
    out["is_millionaire_cliff"] = (income_int >= 170537).astype("int8")
    out["is_dead_zone"] = ((income_int >= 31004) & (income_int <= 41970)).astype("int8")
    out["is_env_hater"] = (env == 1).astype("int8")
    
    # Novel ratio/product interactions (NOT yet tried as pure features)
    out["income_per_km"] = np.where(commute > 0, income / commute, 0).astype(np.float32)
    out["income_x_env"] = (income * env / 100000.0).astype(np.float32)
    out["charge_total"] = (charge_home + charge_work).astype(np.float32)
    out["charge_ratio"] = np.where(charge_work > 0, charge_home / charge_work, 0).astype(np.float32)
    out["charge_density"] = (charge_home * charge_work).astype(np.float32)
    out["age_x_income"] = (age * income / 1e6).astype(np.float32)
    out["income_per_age"] = np.where(age > 0, income / age, 0).astype(np.float32)
    out["km_per_charge"] = np.where(charge_home + charge_work > 0, 
                                     commute / (charge_home + charge_work), 0).astype(np.float32)
    out["env_x_charge"] = (env * (charge_home + charge_work)).astype(np.float32)
    out["cars_x_income"] = (cars * income / 100000.0).astype(np.float32)
    out["env_per_cars"] = np.where(cars > 0, env / cars, env).astype(np.float32)
    out["commute_x_charge_home"] = (commute * charge_home).astype(np.float32)
    out["log_income"] = np.log1p(income).astype(np.float32)
    out["income_sq"] = ((income / 100000.0) ** 2).astype(np.float32)
    out["commute_sq"] = (commute ** 2).astype(np.float32)
    
    return out

X = build_features(train.drop(columns=[ID, TARGET]))
X_test = build_features(test.drop(columns=[ID]))
print(f"Features built: {X.shape[1]} ({time.time()-T0:.0f}s)", flush=True)

# CatBoost needs string categoricals
cat_indices = [list(X.columns).index(c) for c in CATS]
for c in CATS:
    X[c] = X[c].astype(str)
    X_test[c] = X_test[c].astype(str)

# ---- Training ----
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
SEEDS = [42, 137, 2024]  # Multi-seed for variance reduction

CBM_PARAMS = dict(
    iterations=4000,
    learning_rate=0.02,
    depth=8,
    l2_leaf_reg=3.0,
    border_count=254,
    bagging_temperature=0.5,
    random_strength=1.0,
    min_data_in_leaf=20,
    grow_policy='SymmetricTree',
    bootstrap_type='Bayesian',
    verbose=0,
    task_type='CPU',
    eval_metric='AUC',
    use_best_model=True,
    od_type='Iter',
    od_wait=200,
)

oof_all_seeds = np.zeros((len(SEEDS), N), dtype=np.float64)
test_all_seeds = np.zeros((len(SEEDS), len(test)), dtype=np.float64)

for si, seed in enumerate(SEEDS):
    oof_seed = np.zeros(N, dtype=np.float64)
    test_seed = np.zeros(len(test), dtype=np.float64)
    params = CBM_PARAMS.copy()
    params['random_seed'] = seed
    
    for fold, (tr_idx, va_idx) in enumerate(cv.split(X, y)):
        tf = time.time()
        train_pool = Pool(X.iloc[tr_idx], y[tr_idx], cat_features=cat_indices)
        valid_pool = Pool(X.iloc[va_idx], y[va_idx], cat_features=cat_indices)
        
        model = CatBoostClassifier(**params)
        model.fit(train_pool, eval_set=valid_pool)
        
        pv = model.predict_proba(X.iloc[va_idx])[:, 1]
        pt = model.predict_proba(X_test)[:, 1]
        oof_seed[va_idx] = pv
        test_seed += pt / 5
        
        auc_f = roc_auc_score(y[va_idx], pv)
        print(f"[seed={seed}] fold {fold}: AUC {auc_f:.6f} trees {model.best_iteration_} ({time.time()-tf:.0f}s)", flush=True)
    
    oof_auc = roc_auc_score(y, oof_seed)
    print(f"[seed={seed}] OOF AUC: {oof_auc:.6f}", flush=True)
    oof_all_seeds[si] = oof_seed
    test_all_seeds[si] = test_seed

# Average across seeds
oof_final = oof_all_seeds.mean(axis=0)
test_final = test_all_seeds.mean(axis=0)
oof_auc_final = roc_auc_score(y, oof_final)
print(f"\n=== MULTI-SEED AVERAGED OOF AUC: {oof_auc_final:.7f} ===", flush=True)

# Save OOF and test predictions
np.save("oof_v71_catboost_multiseed.npy", oof_final)
np.save("test_v71_catboost_multiseed.npy", test_final)
for si, seed in enumerate(SEEDS):
    np.save(f"scratch/oof_v71_cbm_seed{seed}.npy", oof_all_seeds[si])
    np.save(f"scratch/test_v71_cbm_seed{seed}.npy", test_all_seeds[si])

# Also save individual seed OOFs for analysis
print(f"Total time: {time.time()-T0:.0f}s", flush=True)
print("Saved oof_v71_catboost_multiseed.npy and test_v71_catboost_multiseed.npy", flush=True)
