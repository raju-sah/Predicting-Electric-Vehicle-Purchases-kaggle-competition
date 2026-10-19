import csv
import json
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
manifest = json.loads((root/'scratch/verified_ensembles/manifest.json').read_text())
path = root/'exp_log.csv'
with path.open(newline='') as f:
    reader = csv.DictReader(f)
    fields = reader.fieldnames
    rows = list(reader)
existing = {r['ver']: r for r in rows}
for i, r in enumerate(manifest['grid']):
    ver = f'verified5_grid_{i:02d}'
    record = {'ver':ver, 'model':f"{r['method']} memory-LGB6-XGB5 weights {r['weights']}",
              'OOF_AUC':r['pooled_auc'], 'public_LB':'NOT submitted',
              'note':'Fixed seed42 fivefold predictions; grid selection reuses OOF labels; details in verified_ensembles/manifest.json'}
    if ver not in existing:
        rows.append(record)
for r in manifest['candidates']:
    ver = r['version']
    if ver not in existing:
        record = {'ver':ver, 'model':f"Verified fivefold {r['method']} ensemble {r['weights']}",
                  'OOF_AUC':r['pooled_auc'], 'public_LB':'PENDING',
                  'note':'Complete candidate OOF; improves all five folds versus memory; full source provenance recorded'}
        rows.append(record)
        existing[ver] = record
for argument in sys.argv[1:]:
    ver, public = argument.split('=',1)
    existing[ver]['public_LB'] = public
with path.open('w',newline='') as f:
    writer = csv.DictWriter(f,fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
print('Recorded all grid trials and three complete candidates.')
