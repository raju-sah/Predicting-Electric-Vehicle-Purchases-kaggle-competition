import json

for nb_name in ["simple-tabm-baseline-s6e9.ipynb", "simple-catboost-baseline-s6e9.ipynb"]:
    print(f"================== {nb_name} ==================")
    with open(f"scratch/kernels_pulled/yhay81/{nb_name}") as f:
        nb = json.load(f)
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'markdown' or any("stack" in l or "feature" in l or "model" in l or "tabm" in l or "catboost" in l for l in cell['source']):
            print(f"--- Cell {i} [{cell['cell_type']}] ---")
            print("".join(cell['source'])[:1000])
            print("\n")
