import json

for nb_path in [
    "scratch/dariushafshar/s6e9-0-94585-1-of-4-ingredients-carries-it.ipynb",
    "scratch/tamerlanomralinov/s6e9-hirge-net-and-doubly-nested-maxcv-stack.ipynb"
]:
    print(f"\n==================== {nb_path} ====================")
    with open(nb_path) as f:
        nb = json.load(f)
    for cell in nb['cells']:
        src = "".join(cell.get('source', []))
        if cell['cell_type'] == 'markdown' and any(k in src.lower() for k in ["ingredient", "carry", "feature", "0.94", "auc", "stack", "net"]):
            print("MD:", src[:400])
            print("---")
        elif cell['cell_type'] == 'code' and any(k in src for k in ["def ", "features", "XGB", "LGBM", "stack", "blend"]):
            print("CODE:", src[:400])
            print("---")
