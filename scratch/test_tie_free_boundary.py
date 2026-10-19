import pandas as pd
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

train = pd.read_csv('train.csv')
y = (train['Will_Buy_EV'] == 'Yes').astype(int).values
income = train['Annual_Income_USD'].values
oof = pd.read_csv('scratch/datasets/four_views/oof_four_views.csv')['ensemble_ABCD'].values

base_auc = roc_auc_score(y, oof)
print(f"Base AUC: {base_auc:.7f}")

# Continuous ordinal ranks [0, 1]
r = (rankdata(oof, method='ordinal') - 0.5) / len(oof)
print(f"Rank AUC: {roc_auc_score(y, r):.7f}")

# Masks
hi_mask = income >= 170537.0
DEAD_LO, DEAD_HI = 31003.0, 41971.0
dead_mask = (income > DEAD_LO) & (income < DEAD_HI)
normal_mask = (~hi_mask) & (~dead_mask)

# Re-order: Dead Zone at the bottom, Normal in the middle, Cliff at the top!
# Each group internally ranked by ordinal rank
r_dead = rankdata(oof[dead_mask], method='ordinal')
r_normal = rankdata(oof[normal_mask], method='ordinal')
r_hi = rankdata(oof[hi_mask], method='ordinal')

# Concatenate ranks globally:
# dead rows get ranks [1 .. len(dead)]
# normal rows get ranks [len(dead)+1 .. len(dead)+len(normal)]
# hi rows get ranks [len(dead)+len(normal)+1 .. total]

final_r = np.zeros(len(oof), dtype=np.float64)
n_dead = len(r_dead)
n_normal = len(r_normal)
n_hi = len(r_hi)

final_r[dead_mask] = r_dead
final_r[normal_mask] = n_dead + r_normal
final_r[hi_mask] = n_dead + n_normal + r_hi

final_r = (final_r - 0.5) / len(final_r)

opt_auc = roc_auc_score(y, final_r)
print(f"Tie-Free Segmented Boundary AUC: {opt_auc:.7f}")
print(f"Net Gain: {opt_auc - base_auc:+.7f}")
