import json
import os

kernels = [
    "beraterolelk_s6e9-triblend-oof-ensemble-xgb-lgbm-catboost",
    "beraterolelk_s6e9-anchor-3seed-lgbm",
    "beraterolelk_s6e9-rebuilt-anchor-lgbm-original-data-3seed"
]

for kd in kernels:
    print(f"\n==================== {kd} ====================")
    p = f"scratch/kernels/{kd}"
    for f in os.listdir(p):
        if f.endswith(".ipynb"):
            with open(os.path.join(p, f)) as ipf:
                nb = json.load(ipf)
                for i, cell in enumerate(nb.get("cells", [])):
                    if cell.get("cell_type") == "markdown":
                        text = "".join(cell.get("source", []))
                        if any(kw in text.lower() for kw in ["0.946", "cv", "oof", "lb", "seed", "blend", "model"]):
                            print(f"[MD {i}]: {text[:300]}\n---")
                    elif cell.get("cell_type") == "code":
                        text = "".join(cell.get("source", []))
                        if any(kw in text.lower() for kw in ["catboost", "lgb", "xgb", "roc_auc", "rank", "blend", "weights"]):
                            print(f"[CODE {i}]: {text[:300]}\n---")

