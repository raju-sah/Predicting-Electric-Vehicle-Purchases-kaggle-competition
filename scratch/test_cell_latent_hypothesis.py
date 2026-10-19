import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold

train = pd.read_csv("train.csv")
y = (train["Will_Buy_EV"] == "Yes").astype(int).values

# Let's inspect AUC of Annual_Income_USD within the 30 cells:
train["cell"] = train["Environmental_Concern_Level"].astype(str) + "_" + train["Subsidy_Available"].astype(str) + "_" + train["Range_Anxiety_Level"].astype(str)

unique_cells = train["cell"].unique()
print(f"Total cells: {len(unique_cells)}")

cell_stats = []
for cell in sorted(unique_cells):
    sub = train[train["cell"] == cell]
    sub_y = y[train["cell"] == cell]
    pos_rate = sub_y.mean()
    n_pos = sub_y.sum()
    n_neg = len(sub_y) - n_pos
    
    if n_pos > 0 and n_neg > 0:
        auc_income = roc_auc_score(sub_y, sub["Annual_Income_USD"])
        auc_commute = roc_auc_score(sub_y, -sub["Daily_Commute_km"])
        auc_chargers = roc_auc_score(sub_y, sub["Charging_Stations_Near_Home"])
    else:
        auc_income, auc_commute, auc_chargers = np.nan, np.nan, np.nan
        
    cell_stats.append({
        "cell": cell,
        "count": len(sub),
        "pos_rate": pos_rate,
        "auc_income": auc_income,
        "auc_commute": auc_commute,
        "auc_chargers": auc_chargers,
    })

stats_df = pd.DataFrame(cell_stats)
print(stats_df.to_string())

print("\nMean within-cell AUC:")
print("Annual_Income_USD:              ", stats_df["auc_income"].mean())
print("Daily_Commute_km:               ", stats_df["auc_commute"].mean())
print("Charging_Stations_Near_Home:    ", stats_df["auc_chargers"].mean())
