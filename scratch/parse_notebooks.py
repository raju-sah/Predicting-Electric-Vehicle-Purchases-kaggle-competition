import json
import glob

def extract_notebook_highlights(path):
    print(f"\n==================== NOTEBOOK: {path} ====================")
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    
    for cell in nb.get("cells", []):
        cell_type = cell.get("cell_type")
        source = "".join(cell.get("source", []))
        if cell_type == "markdown":
            # print markdown headers or interesting text
            if any(k in source.lower() for k in ["0.94", "score", "ingredient", "feature", "blend", "model", "stack", "trick", "insight", "te", "pruning"]):
                print("\n[MARKDOWN]:")
                print(source[:1000])
        elif cell_type == "code":
            # look for imports, feature definitions, weights, or models
            if any(k in source.lower() for k in ["blend", "weight", "sub[", "auc", "roc_auc", "def ", "class ", "params", "xgb.train", "lgb.train"]):
                print("\n[CODE SNIPPET]:")
                lines = source.split("\n")
                filtered = [l for l in lines if any(w in l.lower() for w in ["blend", "weight", "auc", "params", "sub", "roc_auc", "feature", "def ", "predict"])]
                print("\n".join(filtered[:30]))

for p in sorted(glob.glob("scratch/kernels_pulled/*/*.ipynb")):
    extract_notebook_highlights(p)
