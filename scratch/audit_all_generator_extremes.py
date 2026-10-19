import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

train = pd.read_csv("train.csv")
y = (train["Will_Buy_EV"] == "Yes").astype(int).values

# Let's inspect the OOF predictions of naji_xgb and other models
naji_oof = pd.read_csv("scratch/najiama_out/oof_XGBOOST.csv")["OOF_Pred"].values
auc_base = roc_auc_score(y, naji_oof)
print(f"Base naji_xgb OOF AUC: {auc_base:.6f}")

# Check High Anxiety Zero Zone in train
mask_zero_anx = (train["Range_Anxiety_Level"] == "High") & ~((train["Environmental_Concern_Level"] == 5) & (train["Subsidy_Available"] == "Yes"))
print(f"Train Zero Zone 1 (High Anxiety non-(5,Yes)): {mask_zero_anx.sum()} rows, buyers = {y[mask_zero_anx].sum()}")

# Check Millionaire Cliff in train
mask_millionaire = (train["Annual_Income_USD"] >= 170537)
print(f"Train Millionaire Cliff (income >= 170537): {mask_millionaire.sum()} rows, buyers = {y[mask_millionaire].sum()} / {mask_millionaire.sum()} ({y[mask_millionaire].mean():.4f})")

# Check Lower Dead Zone in train
mask_lower_dead = (train["Annual_Income_USD"] > 31003) & (train["Annual_Income_USD"] < 41971) & (train["Subsidy_Available"] == "No")
print(f"Train Lower Dead Zone (31003 < income < 41971 & Subsidy==No): {mask_lower_dead.sum()} rows, buyers = {y[mask_lower_dead].sum()} / {mask_lower_dead.sum()}")

# Let's check other combinations with 0 buyers
grouped = train.groupby(["Environmental_Concern_Level", "Subsidy_Available", "Range_Anxiety_Level"])["Will_Buy_EV"].agg(["count", lambda x: (x == "Yes").sum(), lambda x: (x == "Yes").mean()]).reset_index()
grouped.columns = ["Concern", "Subsidy", "Anxiety", "Count", "Buyers", "Rate"]

zero_cells = grouped[grouped["Buyers"] == 0]
print(f"\nZero-buyer cells ({len(zero_cells)} cells, total {zero_cells['Count'].sum()} rows):")
print(zero_cells.to_string())

# Find all train rows belonging to any zero-buyer cell
is_zero_cell = pd.Series(False, index=train.index)
for idx, r in zero_cells.iterrows():
    m = (train["Environmental_Concern_Level"] == r["Concern"]) & (train["Subsidy_Available"] == r["Subsidy"]) & (train["Range_Anxiety_Level"] == r["Anxiety"])
    is_zero_cell |= m

print(f"\nTotal train rows in zero-buyer cells: {is_zero_cell.sum()} rows, buyers = {y[is_zero_cell].sum()}")

# Test OOF effect if zero-cells are floored / ranked at bottom
# Since AUC depends on ranking:
# If a row has true y = 0, ranking it as low as possible can only IMPROVE or preserve AUC!
# Let's verify on OOF!
modified_oof = naji_oof.copy()
# Penalize zero-cell rows by shifting their prediction down
modified_oof[is_zero_cell] = modified_oof[is_zero_cell] * 0.01

# Also boost millionaire cliff rows
modified_oof[mask_millionaire] = modified_oof[mask_millionaire] + 10.0

auc_mod = roc_auc_score(y, modified_oof)
print(f"OOF AUC after applying Zero-Cell & Millionaire Cliff constraints: {auc_mod:.7f} (delta = {auc_mod - auc_base:+.7f})")
