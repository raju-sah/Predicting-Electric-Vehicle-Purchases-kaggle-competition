import pandas as pd
import numpy as np

test = pd.read_csv("test.csv")
v11 = pd.read_csv("subs/v11_megayak_4view_daily.csv")

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

p = v11.loc[is_test_zero, "Will_Buy_EV"]
print("v11 predictions on the 3,340 zero-cell rows:")
print(f"  Min:    {p.min():.6f}")
print(f"  25%:    {p.quantile(0.25):.6f}")
print(f"  Median: {p.median():.6f}")
print(f"  75%:    {p.quantile(0.75):.6f}")
print(f"  Max:    {p.max():.6f}")
print(f"  Rows > 0.10: {(p > 0.10).sum()} / {len(p)}")
print(f"  Rows > 0.20: {(p > 0.20).sum()} / {len(p)}")
print(f"  Rows > 0.50: {(p > 0.50).sum()} / {len(p)}")
