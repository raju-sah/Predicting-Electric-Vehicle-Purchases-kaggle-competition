import json

with open("scratch/kernels_pulled/yhay81/simple-tabm-baseline-s6e9.ipynb") as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        print(f"=== CELL {i} ===")
        print("".join(cell['source']))
        print("\n" + "="*50 + "\n")
