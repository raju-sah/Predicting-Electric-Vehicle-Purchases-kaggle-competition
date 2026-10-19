import json

with open("scratch/kernels_pulled/taeyangg4_094650/s6e9-094650-multi-paradigm-lexsort-master.ipynb") as f:
    nb = json.load(f)

print("=== Cell 1 text ===")
print("".join(nb['cells'][1]['source']))
