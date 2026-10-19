import pandas as pd
import numpy as np

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train["cell"] = (
    train["Environmental_Concern_Level"].astype(str)
    + "_"
    + train["Subsidy_Available"].astype(str)
    + "_"
    + train["Range_Anxiety_Level"].astype(str)
)

sub_train = train[train["cell"] == "2.0_No_Medium"]
print("Train rows with 2.0_No_Medium:", len(sub_train))
print("Max income in train for 2.0_No_Medium:", sub_train["Annual_Income_USD"].max())
print("Mean income in train for 2.0_No_Medium:", sub_train["Annual_Income_USD"].mean())
print("Distribution of income for 2.0_No_Medium in train:")
print(sub_train["Annual_Income_USD"].describe())

# Now check row 42 in test:
row42 = test.loc[42]
print("\nRow 42 in test:")
print(row42[["Annual_Income_USD", "Environmental_Concern_Level", "Subsidy_Available", "Range_Anxiety_Level"]])

# Hayashi true formula on row 42:
buy_score_42 = (
    -8.5
    + 0.12 * (row42["Annual_Income_USD"] / 10000.0)
    + 0.6 * row42["Environmental_Concern_Level"]
    + 2.0 * (row42["Subsidy_Available"] == "Yes")
    + 3.0 * (row42["Range_Anxiety_Level"] == "Low")
    + 2.0 * (row42["Range_Anxiety_Level"] == "Medium")
)
print(f"Row 42 Hayashi buy score: {buy_score_42:.4f}")
from scipy.special import ndtr
print(f"Row 42 Probit true buy prob: {ndtr(buy_score_42):.4f}")
