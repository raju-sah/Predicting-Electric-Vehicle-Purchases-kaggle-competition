from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

root = Path(__file__).resolve().parents[1]
train = pd.read_csv(root / 'train.csv', usecols=['id', 'Will_Buy_EV'])
y = (train.Will_Buy_EV == 'Yes').astype(int).to_numpy()
folds = [va for _, va in StratifiedKFold(5, shuffle=True, random_state=42).split(y,y)]
mem = pd.read_csv(root / 'scratch/pull_sep18/memory_recipe_out/oof_predictions.csv').oof_final.to_numpy()
mr = rankdata(mem) / len(mem)
rows = []
for p in sorted((root/'scratch/pull_sep19/najiama_refresh').glob('*oof.csv')):
    df = pd.read_csv(p)
    assert df.id.equals(train.id), p
    col = [c for c in df if c != 'id'][0]
    pred = df[col].to_numpy()
    old = root / 'scratch/pull_sep16/ds_najiama' / p.name
    same = old.exists() and hashlib.sha256(old.read_bytes()).digest() == hashlib.sha256(p.read_bytes()).digest()
    row = {'name': p.name, 'columns': df.columns.tolist(), 'same_as_sep16': same,
           'auc': roc_auc_score(y,pred), 'diagnostic_group_auc': [roc_auc_score(y[v],pred[v]) for v in folds]}
    r = rankdata(pred) / len(pred)
    row['rank_corr_memory'] = float(np.corrcoef(r,mr)[0,1])
    row['memory_half_rank_blend_auc'] = roc_auc_score(y, .5*mr+.5*r)
    rows.append(row)
print(json.dumps(rows, indent=2))
(root/'scratch/refreshed_pool_assessment.json').write_text(json.dumps(rows,indent=2)+'\n')
