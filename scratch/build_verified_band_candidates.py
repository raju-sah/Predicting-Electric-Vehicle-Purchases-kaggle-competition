"""Apply a fixed published public-tuned correction to complete OOF/test pairs."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

root = Path(__file__).resolve().parents[1]
sample = pd.read_csv(root/'sample_submission.csv')
train = pd.read_csv(root/'train.csv',usecols=['Will_Buy_EV'])
y = (train.Will_Buy_EV == 'Yes').to_numpy(np.int8)
folds = [v for _,v in StratifiedKFold(5,shuffle=True,random_state=42).split(y,y)]
bands = [(0.54,0.04),(0.20,0.06),(0.58,0.04),(0.47,0.06)]

def transform(values):
    rank = rankdata(values,method='average')/len(values)
    shifted = rank.copy()
    for lo,width in bands:
        half = width/2
        low = (rank >= lo)&(rank < lo+half)
        high = (rank >= lo+half)&(rank < lo+width)
        shifted[low] += half
        shifted[high] -= half
    return rankdata(shifted,method='ordinal')/len(shifted)

records = []
for version, parent, method in [(69,68,'logit'),(70,67,'rank')]:
    oof = np.load(root/f'scratch/verified_ensembles/oof_v{parent}.npy')
    frame = pd.read_csv(root/f'subs/v{parent}_verified5_{method}.csv')
    assert frame.id.equals(sample.id)
    raw_test = frame.Will_Buy_EV.to_numpy()
    adjusted_oof, adjusted_test = transform(oof), transform(raw_test)
    auc = roc_auc_score(y,adjusted_oof)
    assert auc > .9418+.0005
    frame.Will_Buy_EV = adjusted_test
    filename = f'subs/v{version}_verified5_{method}_bands.csv'
    frame.to_csv(root/filename,index=False)
    np.save(root/f'scratch/verified_ensembles/oof_v{version}.npy',adjusted_oof)
    row = {'version':f'v{version}','parent':f'v{parent}','method':method,'file':filename,
           'pooled_oof_auc':auc,'oof_delta_vs_parent':auc-roc_auc_score(y,oof),
           'fold_auc':[roc_auc_score(y[v],adjusted_oof[v]) for v in folds],
           'bands':bands,'test_rank_correlation_parent':float(spearmanr(adjusted_test,raw_test).statistic),
           'sha256':hashlib.sha256((root/filename).read_bytes()).hexdigest(),
           'caveat':'Full OOF transformation measured; bands were chosen using public leaderboard feedback in prior work. No independent generalization claim.'}
    records.append(row)
    print(json.dumps(row),flush=True)
(root/'scratch/verified_ensembles/band_manifest.json').write_text(json.dumps(records,indent=2)+'\n')
with (root/'exp_log.csv').open(newline='') as f:
    reader = csv.DictReader(f)
    fields = reader.fieldnames
    old = list(reader)
seen = {r['ver'] for r in old}
for r in records:
    if r['version'] not in seen:
        old.append({'ver':r['version'],'model':f"Verified fivefold {r['method']} ensemble plus fixed four-band swaps",
                    'OOF_AUC':r['pooled_oof_auc'],'public_LB':'NOT submitted',
                    'note':r['caveat']})
with (root/'exp_log.csv').open('w',newline='') as f:
    writer = csv.DictWriter(f,fieldnames=fields)
    writer.writeheader()
    writer.writerows(old)
