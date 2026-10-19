import json

print("--- SOUKE AIZEN CELLS 1-5 ---")
with open("scratch/kernels_pulled/souke_v2/s6e9-top-50-formula-dual-master-ran-v2.ipynb") as f:
    nb_souke = json.load(f)

for i in range(min(6, len(nb_souke['cells']))):
    print(f"=== Cell {i} [{nb_souke['cells'][i]['cell_type']}] ===")
    print("".join(nb_souke['cells'][i]['source']))
    print("\n" + "="*50 + "\n")

print("--- NINA H-BLEND 4 CELLS 1-5 ---")
with open("scratch/kernels_pulled/nina_h4/ps-s6e9-h-blend-4.ipynb") as f:
    nb_nina = json.load(f)

for i in range(min(6, len(nb_nina['cells']))):
    print(f"=== Cell {i} [{nb_nina['cells'][i]['cell_type']}] ===")
    print("".join(nb_nina['cells'][i]['source']))
    print("\n" + "="*50 + "\n")
