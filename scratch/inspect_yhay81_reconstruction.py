import json

with open("scratch/kernels_pulled/yhay81/exact-reconstruction-of-the-ev-source-dataset.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

print("=== YUSUKE HAYASHI RECONSTRUCTION NOTEBOOK ===")
for i, cell in enumerate(nb.get("cells", [])):
    cell_type = cell.get("cell_type")
    source = "".join(cell.get("source", []))
    print(f"\n--- CELL {i} [{cell_type}] ---")
    print(source[:1500])
