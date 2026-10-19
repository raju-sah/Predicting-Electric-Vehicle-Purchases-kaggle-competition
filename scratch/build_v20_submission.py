import pandas as pd
import numpy as np
from scipy.stats import rankdata

test = pd.read_csv('test.csv')
sample = pd.read_csv('sample_submission.csv')

v11 = pd.read_csv('subs/v11_megayak_4view_daily.csv')
taeyang = pd.read_csv('scratch/taeyangg4_out/submission.csv')
nina = pd.read_csv('subs/v18_nina_honest_residual_tiebreaker.csv')
deotte = pd.read_csv('scratch/deotte_gpu_out/sub_xgb_deotte_simpson.csv')
realmlp = pd.read_csv('scratch/realmlp_out/submission.csv')

# Verify IDs across all files
for name, df in [('v11', v11), ('taeyang', taeyang), ('nina', nina), ('deotte', deotte), ('realmlp', realmlp)]:
    assert (sample['id'] == df['id']).all(), f"ID mismatch in {name}"

N = len(sample)
r_v11 = (rankdata(v11['Will_Buy_EV'].values, method='ordinal') - 0.5) / N
r_taeyang = (rankdata(taeyang['Will_Buy_EV'].values, method='ordinal') - 0.5) / N
r_nina = (rankdata(nina['Will_Buy_EV'].values, method='ordinal') - 0.5) / N
r_deotte = (rankdata(deotte['Will_Buy_EV'].values, method='ordinal') - 0.5) / N
r_realmlp = (rankdata(realmlp['Will_Buy_EV'].values, method='ordinal') - 0.5) / N

# Weighted ensemble blend
raw_blend = (
    0.40 * r_v11 +
    0.40 * r_taeyang +
    0.10 * r_nina +
    0.06 * r_deotte +
    0.04 * r_realmlp
)

# Apply Tie-Free Segmented Boundary Constraints
income_te = test['Annual_Income_USD'].values

hi_mask = income_te >= 170537.0
DEAD_LO, DEAD_HI = 31003.0, 41971.0
dead_mask = (income_te > DEAD_LO) & (income_te < DEAD_HI)
normal_mask = (~hi_mask) & (~dead_mask)

print(f"Segment breakdown:")
print(f"  Dead zone rows: {dead_mask.sum()}")
print(f"  Normal rows:    {normal_mask.sum()}")
print(f"  Cliff rows:     {hi_mask.sum()}")

r_dead = rankdata(raw_blend[dead_mask], method='ordinal')
r_normal = rankdata(raw_blend[normal_mask], method='ordinal')
r_hi = rankdata(raw_blend[hi_mask], method='ordinal')

final_ranks = np.zeros(N, dtype=np.float64)
n_dead = len(r_dead)
n_normal = len(r_normal)
n_hi = len(r_hi)

final_ranks[dead_mask] = r_dead
final_ranks[normal_mask] = n_dead + r_normal
final_ranks[hi_mask] = n_dead + n_normal + r_hi

final_probs = (final_ranks - 0.5) / N

# Create submission
sub = pd.DataFrame({'id': sample['id'], 'Will_Buy_EV': final_probs})
sub_path = 'subs/v20_boundary_constrained_grandmaster_blend.csv'
sub.to_csv(sub_path, index=False)

# Strict validation
assert len(sub) == 286571, f"Row count mismatch: {len(sub)}"
assert list(sub.columns) == ['id', 'Will_Buy_EV'], f"Columns mismatch: {sub.columns}"
assert (sub['id'] == sample['id']).all(), "ID mismatch"
assert not sub['Will_Buy_EV'].isna().any(), "Found NaNs"
assert np.isfinite(sub['Will_Buy_EV'].values).all(), "Found infinite values"
assert (sub['Will_Buy_EV'] >= 0.0).all() and (sub['Will_Buy_EV'] <= 1.0).all(), "Out of bounds [0, 1]"
assert len(np.unique(final_probs)) == N, "Ties found in final predictions!"

print("v20 created and 100% strictly validated!")
print(f"Spearman corr with v11: {np.corrcoef(final_probs, r_v11)[0, 1]:.7f}")
print(f"Spearman corr with taeyang: {np.corrcoef(final_probs, r_taeyang)[0, 1]:.7f}")
print("Head:")
print(sub.head(5))
