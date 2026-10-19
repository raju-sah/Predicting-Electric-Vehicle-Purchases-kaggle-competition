import json

with open("scratch/cdeotte/fable-5-1-xgb-starter.ipynb") as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    src = "".join(cell.get("source", []))
    if cell["cell_type"] == "code":
        print(f"=== CELL {i} ===")
        print(src)
        print("-----------------------------------")
