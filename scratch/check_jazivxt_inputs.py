import json
import os
import glob

with open("scratch/kernels_pulled/jazivxt_zoom/single-model-zoom-zoom.ipynb") as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    source = "".join(cell['source'])
    if "latest" in source.lower() or "input" in source.lower() or "path" in source.lower():
        print(f"=== CELL {i} [{cell['cell_type']}] ===")
        print(source[:1000])
        print("\n" + "="*50 + "\n")
