"""Select three pooling rules from verified seed-42 five-fold predictions.

Selection scores reuse existing OOF labels and are not an independent final holdout.
No public leaderboard scores are used for weight selection.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'scratch/verified_ensembles'
OUT.mkdir(exist_ok=True)
train = pd.read_csv(ROOT / 'train.csv', usecols=['id', 'Will_Buy_EV'])
sample = pd.read_csv(ROOT / 'sample_submission.csv')
y = (train.Will_Buy_EV == 'Yes').to_numpy(np.int8)
fold = np.full(len(y), -1, np.int8)
for k, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(y, y)):
    fold[va] = k

def read_aligned(path, ids, prediction):
    data = pd.read_csv(ROOT / path)
    assert data.id.equals(ids), path
    values = data[prediction].to_numpy(float)
    assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all(), path
    return values

memory_path = 'scratch/pull_sep18/memory_recipe_out/oof_predictions.csv'
memory = pd.read_csv(ROOT / memory_path)
assert np.array_equal(memory.fold, fold)
assert np.array_equal(memory.Will_Buy_EV, y)
specs = [
    ('memory', memory_path, 'oof_final', 'scratch/pull_sep18/memory_recipe_out/submission_model_only.csv'),
    ('lgb6', 'scratch/pull_sep19/najiama_refresh/Pure LGBM_V6_oof.csv', 'OOF_Pred',
     'scratch/pull_sep19/najiama_refresh/Pure LGBM_V6_test.csv'),
    ('xgb5', 'scratch/pull_sep19/najiama_refresh/XGBoost_Triple_TE_5folds_oof.csv', 'OOF_Pred',
     'scratch/pull_sep19/najiama_refresh/XGBoost_Triple_TE_5folds_test.csv'),
]
oof = [read_aligned(op, train.id, col) for _, op, col, _ in specs]
test = [read_aligned(tp, sample.id, 'Will_Buy_EV') for _, _, _, tp in specs]

# Bind dataset arrays to published notebook outputs whose source/logs show 5 folds.
links = [
    (1, 'scratch/research_next/najiama_latest_output/oof_LIGHTGBM.csv',
     'scratch/research_next/najiama_latest_output/test_LIGHTGBM.csv'),
    (2, 'scratch/research_next/najiama_xgb_output/oof_XGBOOST.csv',
     'scratch/research_next/najiama_xgb_output/test_XGBOOST.csv'),
]
for i, op, tp in links:
    assert np.array_equal(oof[i], read_aligned(op, train.id, 'OOF_Pred')), op
    assert np.array_equal(test[i], read_aligned(tp, sample.id, 'Will_Buy_EV')), tp

def transform(values, method):
    if method == 'rank':
        return (rankdata(values, method='average') - .5) / len(values)
    if method == 'logit':
        return logit(np.clip(values, 1e-7, 1 - 1e-7))
    return values

def blend(arrays, weights, method):
    result = sum(w * a for w, a in zip(weights, arrays))
    return expit(result) if method == 'logit' else result

def score(values):
    scores = [roc_auc_score(y[fold == k], values[fold == k]) for k in range(5)]
    return float(np.mean(scores)), scores

def auc_placements(pred):
    # Per-positive and per-negative AUC contributions, including half-credit ties.
    pos, neg = pred[y == 1], pred[y == 0]
    sorted_pos, sorted_neg = np.sort(pos), np.sort(neg)
    vp = (np.searchsorted(sorted_neg,pos,side='left') + np.searchsorted(sorted_neg,pos,side='right')) / (2*len(neg))
    vn = 1 - (np.searchsorted(sorted_pos,neg,side='left') + np.searchsorted(sorted_pos,neg,side='right')) / (2*len(pos))
    return vp, vn

base_auc = roc_auc_score(y, oof[0])
base_mean, base_folds = score(oof[0])
bp, bn = auc_placements(oof[0])
grid = [(i/5, j/5, (5-i-j)/5) for i in range(1,4) for j in range(1,4) if 5-i-j >= 1]
records, candidates = [], []
for ver, method in [(66, 'probability'), (67, 'rank'), (68, 'logit')]:
    oo = [transform(a, method) for a in oof]
    tt = [transform(a, method) for a in test]
    scored = []
    for weights in grid:
        pred = blend(oo, weights, method)
        mean_auc, per_fold = score(pred)
        pooled = roc_auc_score(y, pred)
        row = {'method': method, 'weights': list(weights), 'pooled_auc': pooled,
               'mean_fold_auc': mean_auc, 'fold_auc': per_fold}
        records.append(row)
        scored.append(row)
    best = max(scored, key=lambda r:r['mean_fold_auc'])
    pred = blend(oo, best['weights'], method)
    test_pred = blend(tt, best['weights'], method)
    pp, pn = auc_placements(pred)
    dp, dn = pp-bp, pn-bn
    se = np.sqrt(dp.var(ddof=1)/len(dp) + dn.var(ddof=1)/len(dn))
    delta = best['pooled_auc']-base_auc
    best.update({'version': f'v{ver}', 'delta_vs_memory': delta,
                 'paired_auc_delta_se': float(se),
                 'descriptive_delta_95ci': [float(delta-1.96*se), float(delta+1.96*se)],
                 'fold_deltas_vs_memory': (np.array(best['fold_auc'])-base_folds).tolist(),
                 'selection_caveat': 'Weights and pooling method assessed on existing OOF; interval does not adjust for selection or training dependence.'})
    assert best['pooled_auc'] > .9418 + .0005
    assert np.isfinite(test_pred).all() and ((test_pred >= 0)&(test_pred <= 1)).all()
    frame = sample.copy()
    frame.Will_Buy_EV = test_pred
    filename = f'subs/v{ver}_verified5_{method}.csv'
    frame.to_csv(ROOT / filename,index=False)
    np.save(OUT / f'oof_v{ver}.npy', pred)
    best['file'] = filename
    best['sha256'] = hashlib.sha256((ROOT / filename).read_bytes()).hexdigest()
    candidates.append(best)
    print(json.dumps(best), flush=True)

manifest = {'components': [x[0] for x in specs], 'outer_split': 'StratifiedKFold(5, shuffle=True, random_state=42)',
            'reference_memory_pooled_auc': base_auc, 'reference_memory_mean_fold_auc': base_mean,
            'grid': records, 'candidates': candidates,
            'submission_order': [c['version'] for c in sorted(candidates,key=lambda r:-r['mean_fold_auc'])]}
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
