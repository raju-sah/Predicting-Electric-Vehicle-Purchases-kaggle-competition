import json
import os

files = [
    "scratch/megayak/the-1000-leak-a-from-scratch-0-946-pipeline.ipynb",
    "scratch/dariushafshar/s6e9-0-938-phenomenon-0-946-generator.ipynb",
    "scratch/beraterolelk/ev-purchase-grandmaster-eda-ensemble.ipynb"
]

for path in files:
    print(f"\n==================== {os.path.basename(path)} ====================")
    with open(path) as f:
        nb = json.load(f)
    for cell in nb['cells']:
        src = "".join(cell.get("source", []))
        if cell["cell_type"] == "markdown" and any(k in src.lower() for k in ["leak", "generator", "feature", "0.946", "cv", "score"]):
            print("MD:", src[:350])
            print("---")
        elif cell["cell_type"] == "code" and any(k in src for k in ["%", "//", "TargetEncoder", "def ", "LGBM", "xgb", "cat", "Pipeline", "blend"]):
            print("CODE:", src[:450])
            print("---")
