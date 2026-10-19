import json

with open("scratch/kernels_pulled/najiama/xgboost-triple-te-dynamic-pruning-lb-0-94639.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    source = "".join(cell.get("source", []))
    if cell.get("cell_type") == "markdown" and any(k in source.lower() for k in ["triple", "pruning", "feature", "score", "lb", "cv", "xgb"]):
        print("[NAJIAMA MD]:", source[:500])
    elif cell.get("cell_type") == "code" and any(k in source.lower() for k in ["target_encode", "triple", "prun", "params", "xgb.train"]):
        print("[NAJIAMA CODE]:")
        for line in source.split("\n")[:25]:
            print("  ", line)
