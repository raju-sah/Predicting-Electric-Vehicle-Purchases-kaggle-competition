import json

with open("scratch/kernels_pulled/georgy/s6e9-what-the-board-paid-for-eleven-submissions.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

for cell in nb.get("cells", []):
    source = "".join(cell.get("source", []))
    if any(k in source for k in ["SUBMISSIONS", "paid", "eleven submissions", "0.946"]):
        print("=== CELL ===")
        print(source[:1200])
