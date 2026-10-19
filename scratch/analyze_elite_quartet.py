import pandas as pd
import numpy as np
from scipy.stats import rankdata

v11 = pd.read_csv('subs/v11_megayak_4view_daily.csv')['Will_Buy_EV'].values
taeyang = pd.read_csv('scratch/taeyangg4_out/submission.csv')['Will_Buy_EV'].values
nina = pd.read_csv('subs/v18_nina_honest_residual_tiebreaker.csv')['Will_Buy_EV'].values
deotte = pd.read_csv('scratch/deotte_gpu_out/sub_xgb_deotte_simpson.csv')['Will_Buy_EV'].values
realmlp = pd.read_csv('scratch/realmlp_out/submission.csv')['Will_Buy_EV'].values

r_v11 = rankdata(v11, method='ordinal') / len(v11)
r_taeyang = rankdata(taeyang, method='ordinal') / len(taeyang)
r_nina = rankdata(nina, method='ordinal') / len(nina)
r_deotte = rankdata(deotte, method='ordinal') / len(deotte)
r_realmlp = rankdata(realmlp, method='ordinal') / len(realmlp)

df_corr = pd.DataFrame({
    'v11': r_v11,
    'taeyang_45': r_taeyang,
    'nina_18': r_nina,
    'deotte_xgb': r_deotte,
    'realmlp': r_realmlp
}).corr(method='spearman')

print("Spearman Rank Correlation Matrix:")
print(df_corr.round(6))
