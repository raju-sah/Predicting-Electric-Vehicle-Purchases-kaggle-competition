import os, time
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

print("Loading data...")
train = pd.read_csv("train.csv")
y = (train["Will_Buy_EV"] == "Yes").astype(int).values
folds = np.load("scratch/golem_lib/folds_seed42.npy")

def engineer_features(df):
    f = df.copy()
    
    # 1. Exact integer-scaled digits
    for col in ["Daily_Commute_km", "Annual_Income_USD"]:
        scaled = np.rint(f[col] * 10000).astype(np.int64)
        for k in [-4, -3, -2, -1, 0, 1, 2, 3]:
            f[f"{col}_dig_{k}"] = (scaled // (10**(k + 4))) % 10
            
    # 2. Simpson interaction key
    f["simpson_key"] = (
        f["Home_Charging_Possible"].astype(str) + "_" +
        f["City_Type"].astype(str) + "_" +
        f["Charging_Stations_Near_Home"].astype(str)
    )
    
    # 3. Ratio and interaction features
    f["public_need_home"] = f["Charging_Stations_Near_Home"] / (1 + (f["Home_Charging_Possible"] == "Yes").astype(int))
    f["income_per_anxiety"] = f["Annual_Income_USD"] / (1 + f["Range_Anxiety_Level"].map({"Low": 0, "Medium": 1, "High": 2}))
    f["concern_x_subsidy"] = f["Environmental_Concern_Level"] * (1 + 2 * (f["Subsidy_Available"] == "Yes").astype(int))
    
    # Convert all string/object columns to category
    for c in f.select_dtypes(include=["object", "string"]).columns:
        f[c] = f[c].astype("category")
        
    cols_to_drop = ["id", "Will_Buy_EV"]
    f = f.drop(columns=[c for c in cols_to_drop if c in f.columns])
    return f

X = engineer_features(train)
features = X.columns.tolist()
print(f"Engineered {len(features)} features: {features[:10]}...")

# 5-fold CV with LightGBM
oof_preds = np.zeros(len(train))

params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_estimators": 1200,
    "verbose": -1,
    "n_jobs": 4,
}

fold_aucs = []
for fold in range(5):
    val_mask = (folds == fold)
    tr_mask = ~val_mask
    
    X_tr = X.iloc[tr_mask].copy()
    X_va = X.iloc[val_mask].copy()
    y_tr, y_va = y[tr_mask], y[val_mask]
    
    # Simpson target encoding inside fold
    means = pd.Series(y_tr).groupby(X_tr["simpson_key"].values).mean()
    global_mean = y_tr.mean()
    
    X_tr["simpson_te"] = X_tr["simpson_key"].map(means).fillna(global_mean).values.astype(np.float32)
    X_va["simpson_te"] = X_va["simpson_key"].map(means).fillna(global_mean).values.astype(np.float32)
    
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
    )
    
    va_pred = model.predict_proba(X_va)[:, 1]
    oof_preds[val_mask] = va_pred
    auc = roc_auc_score(y_va, va_pred)
    fold_aucs.append(auc)
    print(f"Fold {fold}: AUC = {auc:.6f}")

total_auc = roc_auc_score(y, oof_preds)
print(f"\n==========================================")
print(f"LGBM + Simpson TE 5-Fold OOF AUC: {total_auc:.6f}")
print(f"Mean Fold AUC: {np.mean(fold_aucs):.6f} (+/- {np.std(fold_aucs):.6f})")
print(f"==========================================")

# Save OOF
np.save("scratch/oof_lgb_simpson.npy", oof_preds)
print("Saved scratch/oof_lgb_simpson.npy")
