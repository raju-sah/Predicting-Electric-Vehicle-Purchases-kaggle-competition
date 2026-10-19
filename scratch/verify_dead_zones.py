import pandas as pd
import numpy as np

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

y = (train['Will_Buy_EV'] == 'Yes').astype(int).values
income_tr = train['Annual_Income_USD'].values
income_te = test['Annual_Income_USD'].values

# Check millionaire cliff
hi_tr = income_tr >= 170537
hi_te = income_te >= 170537
print(f"Millionaire Cliff (>= 170537):")
print(f"  Train: count={hi_tr.sum()}, buyers={y[hi_tr].sum()}, rate={y[hi_tr].mean():.4f}")
print(f"  Test:  count={hi_te.sum()}")

# Check lower dead zone
DEAD_LO, DEAD_HI = 31003.0, 41971.0
dead_tr = (income_tr > DEAD_LO) & (income_tr < DEAD_HI)
dead_te = (income_te > DEAD_LO) & (income_te < DEAD_HI)
print(f"\nLower Dead Zone ({DEAD_LO} to {DEAD_HI}):")
print(f"  Train: count={dead_tr.sum()}, buyers={y[dead_tr].sum()}, rate={y[dead_tr].mean():.4f}")
print(f"  Test:  count={dead_te.sum()}")

# Check other zero or 100% zones in train!
# Let's inspect income quantiles or min/max
print("\nLowest income buyers:")
buyers = train[y == 1]
non_buyers = train[y == 0]
print(f"Min buyer income: {buyers['Annual_Income_USD'].min()}")
print(f"Max non-buyer income: {non_buyers['Annual_Income_USD'].max()}")

# Look at distribution of buyers below 42000
low_inc_buyers = train[(train['Annual_Income_USD'] < 45000) & (y == 1)]
print(f"Total buyers with income < 45,000: {len(low_inc_buyers)}")
print("Income values of buyers < 45,000:")
print(low_inc_buyers['Annual_Income_USD'].value_counts().head(10))
