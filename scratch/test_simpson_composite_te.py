import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import TargetEncoder
import lightgbm as lgb

train = pd.read_csv("train.csv")
y = (train["Will_Buy_EV"] == "Yes").astype(int).values

train["simpson_key"] = (
    train["Home_Charging_Possible"].astype(str) + "_" +
    train["City_Type"].astype(str) + "_" +
    train["Charging_Stations_Near_Home"].astype(str)
)

print(f"Total unique simpson keys: {train['simpson_key'].nunique()}")

# Calculate stats per key
stats = train.groupby("simpson_key")["Will_Buy_EV"].agg(["count", lambda x: (x == "Yes").mean()]).reset_index()
stats.columns = ["simpson_key", "count", "ev_rate"]
stats = stats.sort_values("ev_rate", ascending=False)
print("\nTop 10 highest EV rate keys:")
print(stats.head(10).to_string(index=False))
print("\nTop 10 lowest EV rate keys:")
print(stats.tail(10).to_string(index=False))

# Check the spread of EV rates across these 90 keys!
print(f"\nEV rate range across keys: min = {stats['ev_rate'].min():.4f}, max = {stats['ev_rate'].max():.4f}, std = {stats['ev_rate'].std():.4f}")

# Check correlation of out-of-fold target encoding with target
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_te = np.zeros(len(train))

for tr_idx, va_idx in skf.split(train, y):
    tr_df, va_df = train.iloc[tr_idx], train.iloc[va_idx]
    means = tr_df.groupby("simpson_key")["Will_Buy_EV"].apply(lambda x: (x == "Yes").mean())
    global_mean = (tr_df["Will_Buy_EV"] == "Yes").mean()
    oof_te[va_idx] = va_df["simpson_key"].map(means).fillna(global_mean).values

te_auc = roc_auc_score(y, oof_te)
print(f"\nOOF AUC of Simpson Key Target Encoding ALONE: {te_auc:.6f}")
