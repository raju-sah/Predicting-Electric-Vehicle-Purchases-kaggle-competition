import json

with open("scratch/beraterolelk/ev-purchase-grandmaster-eda-ensemble.ipynb") as f:
    nb = json.load(f)

print("--- Cell 6 ---")
print("".join(nb['cells'][6]['source']))
print("--- Cell 10 ---")
print("".join(nb['cells'][10]['source']))
