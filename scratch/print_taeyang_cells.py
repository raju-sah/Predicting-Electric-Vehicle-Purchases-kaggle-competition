import json

with open("scratch/kernels_pulled/taeyangg4_094650/s6e9-094650-multi-paradigm-lexsort-master.ipynb") as f:
    nb = json.load(f)

for i in [6, 7, 8, 9, 10, 11]:
    if i < len(nb['cells']):
        print(f"=== CELL {i} [{nb['cells'][i]['cell_type']}] ===")
        print("".join(nb['cells'][i]['source']))
        print("\n" + "="*50 + "\n")
