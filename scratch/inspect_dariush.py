import json

with open("scratch/kernels_pulled/dariush/s6e9-0-94585-1-of-4-ingredients-carries-it.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    source = "".join(cell.get("source", []))
    if cell.get("cell_type") == "markdown":
        print("[DARIUSH MD]:", source[:400])
    elif cell.get("cell_type") == "code" and any(k in source.lower() for k in ["blend", "ingredient", "auc", "roc_auc", "weight"]):
        print("[DARIUSH CODE]:")
        for line in source.split("\n")[:20]:
            print("  ", line)
