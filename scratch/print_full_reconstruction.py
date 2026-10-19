import json

with open("scratch/kernels_pulled/yhay81/exact-reconstruction-of-the-ev-source-dataset.ipynb") as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    print(f"=== CELL {i} [{cell['cell_type']}] ===")
    print("".join(cell['source']))
    print("\n" + "="*50 + "\n")
