import json

with open("scratch/nina2025/ps-s6e9-honest-round.ipynb") as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        print("".join(cell.get('source', [])))
        print("---")
