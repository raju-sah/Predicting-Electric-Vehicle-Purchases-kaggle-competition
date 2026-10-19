import json
import pandas as pd
import numpy as np

print("--- Inspecting Taeyang 0.94650 notebook ---")
with open("scratch/kernels_pulled/taeyangg4_094650/s6e9-094650-multi-paradigm-lexsort-master.ipynb") as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'markdown':
        print(f"=== CELL {i} [markdown] ===")
        print("".join(cell['source'])[:800])
        print("\n")
    elif cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if any(w in source.lower() for w in ["blend", "lexsort", "weight", "auc", "submission", "rank"]):
            print(f"=== CELL {i} [code] ===")
            print(source[:1000])
            print("\n")

# Check submission.csv properties
sub = pd.read_csv("scratch/kernels_pulled/taeyangg4_094650_out/submission.csv")
print("Submission shape:", sub.shape)
print("Submission stats:")
print(sub["Will_Buy_EV"].describe())
