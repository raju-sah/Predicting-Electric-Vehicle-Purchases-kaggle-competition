import json

with open("scratch/kernels_pulled/taeyangg4_094650/s6e9-094650-multi-paradigm-lexsort-master.ipynb") as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    source = "".join(cell['source'])
    if any(k.lower() in source.lower() for k in ["quadstack", "exp055", "regime", "infrastructure"]):
        print(f"=== Cell {i} [{cell['cell_type']}] ===")
        print(source[:1500])
        print("\n" + "="*50 + "\n")
