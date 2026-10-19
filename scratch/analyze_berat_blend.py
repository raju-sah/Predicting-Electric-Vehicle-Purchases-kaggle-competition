import pandas as pd
import numpy as np
from scipy.stats import rankdata, spearmanr

v11 = pd.read_csv('subs/v11_megayak_4view_daily.csv')['Will_Buy_EV'].values
deotte = pd.read_csv('scratch/deotte_gpu_out/sub_xgb_deotte_simpson.csv')['Will_Buy_EV'].values
realmlp = pd.read_csv('scratch/realmlp_out/submission.csv')['Will_Buy_EV'].values
berat = pd.read_csv('scratch/beraterolelk_out/submission.csv')['Will_Buy_EV'].values

r_v11 = rankdata(v11) / len(v11)
r_deotte = rankdata(deotte) / len(deotte)
r_realmlp = rankdata(realmlp) / len(realmlp)
r_berat = rankdata(berat) / len(berat)

df_corr = pd.DataFrame({
    'v11': r_v11,
    'deotte_xgb': r_deotte,
    'realmlp': r_realmlp,
    'berat_lgb': r_berat
}).corr(method='spearman')

print("Spearman Rank Correlation Matrix:")
print(df_corr.round(5))
