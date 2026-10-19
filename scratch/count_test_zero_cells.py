import pandas as pd
import numpy as np

test = pd.read_csv("test.csv")

zero_cell_tuples = [
    (1.0, "No", "High"),
    (1.0, "Yes", "High"),
    (2.0, "No", "High"),
    (2.0, "No", "Medium"),
    (2.0, "Yes", "High"),
    (3.0, "No", "High"),
    (3.0, "Yes", "High"),
    (4.0, "No", "High"),
    (4.0, "Yes", "High"),
    (5.0, "No", "High"),
]

is_test_zero = pd.Series(False, index=test.index)
for c, s, a in zero_cell_tuples:
    m = (test["Environmental_Concern_Level"] == c) & (test["Subsidy_Available"] == s) & (test["Range_Anxiety_Level"] == a)
    is_test_zero |= m

print(f"Total test rows in these 10 zero-buyer cells: {is_test_zero.sum():,} ({is_test_zero.mean():.4%})")

# Check millionaire cliff in test
mask_test_millionaire = (test["Annual_Income_USD"] >= 170537)
print(f"Total test rows in Millionaire Cliff (income >= 170537): {mask_test_millionaire.sum():,}")
