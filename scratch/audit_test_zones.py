import pandas as pd
import numpy as np

test = pd.read_csv('test.csv')
v11 = pd.read_csv('subs/v11_megayak_4view_daily.csv')

income_te = test['Annual_Income_USD'].values
p_v11 = v11['Will_Buy_EV'].values

# Millionaire cliff in test
te_hi = income_te >= 170537
print(f"Millionaire Cliff (>= 170537) in Test (156 rows):")
print(f"  Min prediction in v11:  {p_v11[te_hi].min():.6f}")
print(f"  Mean prediction in v11: {p_v11[te_hi].mean():.6f}")
print(f"  Rows with p < 0.999:    {(p_v11[te_hi] < 0.999).sum()} / {te_hi.sum()}")
print(f"  Lowest 5 predictions:\n{sorted(p_v11[te_hi])[:5]}")

# Lower dead zone in test
DEAD_LO, DEAD_HI = 31003.0, 41971.0
te_dead = (income_te > DEAD_LO) & (income_te < DEAD_HI)
print(f"\nLower Dead Zone (31003 to 41971) in Test (494 rows):")
print(f"  Max prediction in v11:  {p_v11[te_dead].max():.6f}")
print(f"  Mean prediction in v11: {p_v11[te_dead].mean():.6f}")
print(f"  Rows with p > 0.05:     {(p_v11[te_dead] > 0.05).sum()} / {te_dead.sum()}")
print(f"  Highest 5 predictions:\n{sorted(p_v11[te_dead], reverse=True)[:5]}")
