import json
import pandas as pd
import numpy as np

print("=== NINA H-BLEND 4 ===")
with open("scratch/kernels_pulled/nina_h4/ps-s6e9-h-blend-4.ipynb") as f:
    nb_nina = json.load(f)

for i, cell in enumerate(nb_nina['cells']):
    source = "".join(cell['source'])
    if cell['cell_type'] == 'markdown' or any(w in source.lower() for w in ["blend", "score", "sub", "model", "auc"]):
        print(f"--- Nina Cell {i} [{cell['cell_type']}] ---")
        print(source[:1200])
        print("\n")

print("\n=== SOUKE AIZEN V2 ===")
with open("scratch/kernels_pulled/souke_v2/s6e9-top-50-formula-dual-master-ran-v2.ipynb") as f:
    nb_souke = json.load(f)

for i, cell in enumerate(nb_souke['cells']):
    source = "".join(cell['source'])
    if cell['cell_type'] == 'markdown' or any(w in source.lower() for w in ["blend", "formula", "sub", "weight"]):
        print(f"--- Souke Cell {i} [{cell['cell_type']}] ---")
        print(source[:1200])
        print("\n")
