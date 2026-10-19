"""Recompute evidence without fitting models or using leaderboard labels."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

root = Path(__file__).resolve().parents[1]
train = pd.read_csv(root / 'train.csv', usecols=['id', 'Will_Buy_EV'])
y = (train.Will_Buy_EV == 'Yes').astype(np.int8).to_numpy()
fold = np.full(len(y), -1, dtype=np.int8)
for k, (_, va) in enumerate(StratifiedKFold(5, shuffle=True, random_state=42).split(np.zeros(len(y)), y)):
    fold[va] = k

report = {'threshold': 0.9423, 'sources': {}, 'metrics': {}}
sources = {
    'memory': 'scratch/pull_sep18/memory_recipe_out/oof_predictions.csv',
    'six_views': 'scratch/pull_sep16/ds_megayak6/oof_six_views.csv',
}
for name, path in sources.items():
    data = pd.read_csv(root / path)
    aligned = data.id.equals(train.id)
    labels = bool(np.array_equal(data.Will_Buy_EV.to_numpy(), y))
    report['sources'][name] = {
        'row_count': len(data), 'id_order_matches': aligned,
        'labels_match': labels, 'folds_present': sorted(data.fold.unique().tolist()),
        'fold_assignment_matches_seed42_5fold': bool(np.array_equal(data.fold.to_numpy(), fold)),
    }
    if name == 'memory':
        assert aligned and labels and np.array_equal(data.fold.to_numpy(), fold)
        for col in ['oof_plain', 'oof_residual', 'oof_final']:
            pred = data[col].to_numpy()
            aucs = [roc_auc_score(y[fold == k], pred[fold == k]) for k in range(5)]
            report['metrics'][col] = {'pooled_auc': roc_auc_score(y, pred), 'fold_auc': aucs,
                                      'mean_fold_auc': float(np.mean(aucs))}

for name in ['v37b_stack', 'v56_memory_stack']:
    pred = np.load(root / f'oof_{name}.npy')
    report['metrics'][name] = {
        'pooled_auc': roc_auc_score(y, pred),
        'auc_on_seed42_row_groups': [roc_auc_score(y[fold == k], pred[fold == k]) for k in range(5)],
        'caution': 'Row-group scoring does not establish component training-fold provenance or independent ensemble selection.'
    }

report['submission_evidence'] = {
    'v57': 'Public score 0.94656; no matching full-submission OOF artifact established.',
    'v58_v59': 'Memory component OOF is available; complete blended-submission OOF is not established.',
    'guaranteed_first_place_candidates': 0,
}
out = root / 'scratch/submission_evidence_audit.json'
out.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
