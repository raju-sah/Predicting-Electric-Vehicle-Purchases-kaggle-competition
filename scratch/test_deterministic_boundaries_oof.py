import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

train = pd.read_csv('train.csv')
y = (train['Will_Buy_EV'] == 'Yes').astype(int).values
income = train['Annual_Income_USD'].values

# Load our best OOF predictions (Four-Views or GPU XGB)
oof = pd.read_csv('scratch/datasets/four_views/oof_four_views.csv')['ensemble_ABCD'].values
base_auc = roc_auc_score(y, oof)
print(f"Base OOF AUC: {base_auc:.7f}")

# 1. Test Millionaire Cliff
hi_mask = income >= 170537.0
oof_hi = oof.copy()
# Set cliff rows to maximum prediction (or push them to top)
max_val = oof.max() + 0.01
oof_hi[hi_mask] = max_val
auc_hi = roc_auc_score(y, oof_hi)
print(f"OOF AUC after Upper Cliff Fix: {auc_hi:.7f} (Delta: {auc_hi - base_auc:+.7f})")

# 2. Test Lower Dead Zone
DEAD_LO, DEAD_HI = 31003.0, 41971.0
dead_mask = (income > DEAD_LO) & (income < DEAD_HI)
oof_dead = oof.copy()
# Set dead zone rows to minimum prediction
min_val = oof.min() - 0.01
oof_dead[dead_mask] = min_val
auc_dead = roc_auc_score(y, oof_dead)
print(f"OOF AUC after Lower Dead Zone Fix: {auc_dead:.7f} (Delta: {auc_dead - base_auc:+.7f})")

# 3. Both together
oof_both = oof.copy()
oof_both[hi_mask] = max_val
oof_both[dead_mask] = min_val
auc_both = roc_auc_score(y, oof_both)
print(f"OOF AUC after BOTH Fixes: {auc_both:.7f} (Delta: {auc_both - base_auc:+.7f})")
