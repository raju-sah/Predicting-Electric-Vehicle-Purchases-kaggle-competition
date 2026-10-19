import json

with open("scratch/beraterolelk/ev-purchase-grandmaster-eda-ensemble.ipynb") as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    src = "".join(cell.get("source", []))
    if any(k in src for k in ["FEAT", "feature", "round", "digit", "TargetEncoder", "params"]):
        print(f"--- Cell {i} ({cell['cell_type']}) ---")
        print(src[:600])
        print()
