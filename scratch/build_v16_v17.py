import pandas as pd
import numpy as np
from scipy.stats import rankdata, norm

sample = pd.read_csv('sample_submission.csv')
v11 = pd.read_csv('subs/v11_megayak_4view_daily.csv')
berat = pd.read_csv('scratch/beraterolelk_out/submission.csv')
deotte = pd.read_csv('scratch/deotte_gpu_out/sub_xgb_deotte_simpson.csv')
realmlp = pd.read_csv('scratch/realmlp_out/submission.csv')

# Verify IDs across all four files
assert (sample['id'] == v11['id']).all()
assert (sample['id'] == berat['id']).all()
assert (sample['id'] == deotte['id']).all()
assert (sample['id'] == realmlp['id']).all()

N = len(sample)
r_v11 = (rankdata(v11['Will_Buy_EV'].values) - 0.5) / N
r_berat = (rankdata(berat['Will_Buy_EV'].values) - 0.5) / N
r_deotte = (rankdata(deotte['Will_Buy_EV'].values) - 0.5) / N
r_realmlp = (rankdata(realmlp['Will_Buy_EV'].values) - 0.5) / N

# Probit transformation (Rank-Gauss)
z_v11 = norm.ppf(np.clip(r_v11, 1e-7, 1.0 - 1e-7))
z_berat = norm.ppf(np.clip(r_berat, 1e-7, 1.0 - 1e-7))
z_deotte = norm.ppf(np.clip(r_deotte, 1e-7, 1.0 - 1e-7))
z_realmlp = norm.ppf(np.clip(r_realmlp, 1e-7, 1.0 - 1e-7))

# --- Submission 1: v16 Quad-Blend (55 / 20 / 15 / 10) ---
z_v16 = 0.55 * z_v11 + 0.20 * z_berat + 0.15 * z_deotte + 0.10 * z_realmlp
p_v16 = norm.cdf(z_v16)

sub_v16 = pd.DataFrame({'id': sample['id'], 'Will_Buy_EV': p_v16})
sub_v16_path = 'subs/v16_grandmaster_probit_quad.csv'
sub_v16.to_csv(sub_v16_path, index=False)

# Validation tests for v16
assert len(sub_v16) == 286571
assert list(sub_v16.columns) == ['id', 'Will_Buy_EV']
assert (sub_v16['id'] == sample['id']).all()
assert not sub_v16['Will_Buy_EV'].isna().any()
assert np.isfinite(sub_v16['Will_Buy_EV'].values).all()
assert (sub_v16['Will_Buy_EV'] >= 0.0).all() and (sub_v16['Will_Buy_EV'] <= 1.0).all()

# --- Submission 2: v17 High-Diversity Probit Blend (45 / 25 / 15 / 15) ---
z_v17 = 0.45 * z_v11 + 0.25 * z_berat + 0.15 * z_deotte + 0.15 * z_realmlp
p_v17 = norm.cdf(z_v17)

sub_v17 = pd.DataFrame({'id': sample['id'], 'Will_Buy_EV': p_v17})
sub_v17_path = 'subs/v17_high_diversity_probit_blend.csv'
sub_v17.to_csv(sub_v17_path, index=False)

# Validation tests for v17
assert len(sub_v17) == 286571
assert list(sub_v17.columns) == ['id', 'Will_Buy_EV']
assert (sub_v17['id'] == sample['id']).all()
assert not sub_v17['Will_Buy_EV'].isna().any()
assert np.isfinite(sub_v17['Will_Buy_EV'].values).all()
assert (sub_v17['Will_Buy_EV'] >= 0.0).all() and (sub_v17['Will_Buy_EV'] <= 1.0).all()

print("v16 and v17 created and verified successfully!")
print(f"v16 Spearman corr with v11: {np.corrcoef(rankdata(p_v16), rankdata(v11['Will_Buy_EV'].values))[0, 1]:.6f}")
print(f"v17 Spearman corr with v11: {np.corrcoef(rankdata(p_v17), rankdata(v11['Will_Buy_EV'].values))[0, 1]:.6f}")
print(f"v16 vs v17 Spearman corr:  {np.corrcoef(rankdata(p_v16), rankdata(p_v17))[0, 1]:.6f}")
