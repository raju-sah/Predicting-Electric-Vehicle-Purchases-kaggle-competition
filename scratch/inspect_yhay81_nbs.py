import json
import glob

for nb_path in sorted(glob.glob("scratch/kernels_pulled/yhay81/*.ipynb")):
    print(f"\n==================== NOTEBOOK: {nb_path} ====================")
    with open(nb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    for cell in nb.get("cells", []):
        source = "".join(cell.get("source", []))
        if cell.get("cell_type") == "markdown" and any(k in source.lower() for k in ["cv", "lb", "reconstruct", "tabm", "catboost", "feature"]):
            print("[MD]:", source[:500])
        elif cell.get("cell_type") == "code" and any(k in source.lower() for k in ["def ", "class ", "params", "catboost", "tabm", "roc_auc", "sub"]):
            lines = [l for l in source.split("\n") if any(k in l.lower() for k in ["def ", "tabm", "catboost", "model", "auc", "roc_auc", "weight", "sub"])]
            if lines:
                print("[CODE]:", "\n".join(lines[:10]))
