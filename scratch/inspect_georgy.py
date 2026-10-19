import json

with open("scratch/kernels_pulled/georgy/s6e9-what-the-board-paid-for-eleven-submissions.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    source = "".join(cell.get("source", []))
    if cell.get("cell_type") == "markdown":
        lines = [l for l in source.split("\n") if any(k in l.lower() for k in ["sub", "lb", "cv", "auc", "paid", "gain", "0.94", "rank", "lesson", "table", "model"])]
        if lines:
            print("[GEORGY MD]:\n" + "\n".join(lines[:15]))
    elif cell.get("cell_type") == "code":
        lines = [l for l in source.split("\n") if any(k in l.lower() for k in ["sub", "auc", "weight", "roc_auc", "rank", "blend", "model"])]
        if len(lines) > 3:
            print("[GEORGY CODE]:\n" + "\n".join(lines[:15]))
