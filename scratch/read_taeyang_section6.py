import json

with open("scratch/taeyangg4/s6e9-094645-multi-paradigm-stacking.ipynb") as f:
    nb = json.load(f)

for cell in nb['cells']:
    src = "".join(cell.get('source', []))
    if "Section 6" in src or "The Road to 0.94646" in src or "tr_hi" in src or "Millionaire" in src or "Dead Zone" in src:
        print(src)
        print("-----------------------------------------")
