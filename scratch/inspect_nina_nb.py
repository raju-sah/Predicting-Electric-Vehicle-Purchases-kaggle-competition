import json

with open("scratch/kernels_pulled/nina/ps-s6e9-ensemble-new-engine-1.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    source = "".join(cell.get("source", []))
    print(f"[{cell.get('cell_type')}]:")
    print(source)
    print("="*40)
