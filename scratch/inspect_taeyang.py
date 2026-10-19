import json

with open("scratch/taeyangg4/s6e9-094645-multi-paradigm-stacking.ipynb") as f:
    nb = json.load(f)

for cell in nb['cells']:
    src = "".join(cell.get('source', []))
    if cell['cell_type'] == 'code':
        print("CODE:", src[:500])
        print("---")
    elif cell['cell_type'] == 'markdown' and any(k in src.lower() for k in ["stack", "blend", "model", "cv", "lb", "weight"]):
        print("MD:", src[:400])
        print("---")
