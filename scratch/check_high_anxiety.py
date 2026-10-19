import pandas as pd
import numpy as np

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

y = (train["Will_Buy_EV"] == "Yes").astype(int).values

mask_train_high = (train["Range_Anxiety_Level"] == "High") & ~((train["Environmental_Concern_Level"] == 5) & (train["Subsidy_Available"] == "Yes"))
print(f"Train high anxiety (excluding Concern 5 & Subsidy Yes): {mask_train_high.sum()} rows, buyers: {y[mask_train_high].sum()}")

mask_test_high = (test["Range_Anxiety_Level"] == "High") & ~((test["Environmental_Concern_Level"] == 5) & (test["Subsidy_Available"] == "Yes"))
print(f"Test matching rows: {mask_test_high.sum()} rows")

# Check income distribution of the 3 buyers in Concern 5 & Subsidy Yes & High Anxiety:
sub_5_yes_high = train[(train["Range_Anxiety_Level"] == "High") & (train["Environmental_Concern_Level"] == 5) & (train["Subsidy_Available"] == "Yes")]
print(f"\nConcern 5 & Subsidy Yes & High Anxiety: {len(sub_5_yes_high)} rows, buyers: {y[(train['Range_Anxiety_Level'] == 'High') & (train['Environmental_Concern_Level'] == 5) & (train['Subsidy_Available'] == 'Yes')].sum()}")
print("Incomes of the buyers:")
print(sub_5_yes_high[y[(train['Range_Anxiety_Level'] == 'High') & (train['Environmental_Concern_Level'] == 5) & (train['Subsidy_Available'] == 'Yes')] == 1]["Annual_Income_USD"].values)
print("Incomes of the non-buyers in this cell:")
print(sub_5_yes_high[y[(train['Range_Anxiety_Level'] == 'High') & (train['Environmental_Concern_Level'] == 5) & (train['Subsidy_Available'] == 'Yes')] == 0]["Annual_Income_USD"].describe())
