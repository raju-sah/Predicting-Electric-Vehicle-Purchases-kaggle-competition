import json

with open("scratch/kernels_pulled/souke_v2/s6e9-top-50-formula-dual-master-ran-v2.ipynb") as f:
    nb_souke = json.load(f)

for i in range(5):
    print(f"=== Souke Cell {i} ===")
    print("".join(nb_souke['cells'][i]['source']))
