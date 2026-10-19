import pandas as pd
import numpy as np
from scipy.stats import rankdata

sample = pd.read_csv('sample_submission.csv')
v11 = pd.read_csv('subs/v11_megayak_4view_daily.csv')
nina = pd.read_csv('scratch/nina2025_out/submission.csv')
deotte = pd.read_csv('scratch/deotte_gpu_out/sub_xgb_deotte_simpson.csv')
realmlp = pd.read_csv('scratch/realmlp_out/submission.csv')

# Verify matching IDs
for df in [v11, nina, deotte, realmlp]:
    assert (sample['id'] == df['id']).all()

# --- v18: Nina's Direct Tie-Breaker ---
sub_v18 = pd.DataFrame({'id': sample['id'], 'Will_Buy_EV': nina['Will_Buy_EV'].values})
sub_v18_path = 'subs/v18_nina_honest_residual_tiebreaker.csv'
sub_v18.to_csv(sub_v18_path, index=False)

assert len(sub_v18) == 286571
assert list(sub_v18.columns) == ['id', 'Will_Buy_EV']
assert (sub_v18['id'] == sample['id']).all()
assert not sub_v18['Will_Buy_EV'].isna().any()
assert np.isfinite(sub_v18['Will_Buy_EV'].values).all()
assert (sub_v18['Will_Buy_EV'] >= 0.0).all() and (sub_v18['Will_Buy_EV'] <= 1.0).all()

# --- v19: Ultra-Conservative Micro-Blend ---
r_v11 = (rankdata(v11['Will_Buy_EV'].values) - 0.5) / len(v11)
r_nina = (rankdata(nina['Will_Buy_EV'].values) - 0.5) / len(nina)
r_deotte = (rankdata(deotte['Will_Buy_EV'].values) - 0.5) / len(deotte)
r_realmlp = (rankdata(realmlp['Will_Buy_EV'].values) - 0.5) / len(realmlp)

blend_19 = 0.88 * r_v11 + 0.06 * r_nina + 0.04 * r_deotte + 0.02 * r_realmlp
r_blend_19 = (rankdata(blend_19) - 0.5) / len(blend_19)

sub_v19 = pd.DataFrame({'id': sample['id'], 'Will_Buy_EV': r_blend_19})
sub_v19_path = 'subs/v19_ultra_conservative_micro_blend.csv'
sub_v19.to_csv(sub_v19_path, index=False)

assert len(sub_v19) == 286571
assert list(sub_v19.columns) == ['id', 'Will_Buy_EV']
assert (sub_v19['id'] == sample['id']).all()
assert not sub_v19['Will_Buy_EV'].isna().any()
assert np.isfinite(sub_v19['Will_Buy_EV'].values).all()
assert (sub_v19['Will_Buy_EV'] >= 0.0).all() and (sub_v19['Will_Buy_EV'] <= 1.0).all()

print("v18 and v19 prepared and validated successfully!")
print(f"v18 head:\n{sub_v18.head(3)}")
print(f"v19 head:\n{sub_v19.head(3)}")
