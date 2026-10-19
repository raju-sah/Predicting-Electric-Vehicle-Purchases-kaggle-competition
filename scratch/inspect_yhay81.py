import json

for nb_name in ["exact-reconstruction-of-the-ev-source-dataset.ipynb", "simple-tabm-baseline-s6e9.ipynb", "simple-catboost-baseline-s6e9.ipynb"]:
    path = f"scratch/yhay81/{nb_name}"
    print(f"\n==================== {nb_name} ====================")
    with open(path) as f:
        nb = json.load(f)
    for cell in nb['cells']:
        src = "".join(cell.get("source", []))
        if cell["cell_type"] == "markdown" and any(k in src.lower() for k in ["cv", "score", "feature", "model", "auc"]):
            print("MD:", src[:300])
        elif cell["cell_type"] == "code" and any(k in src for k in ["def ", "TargetEncoder", "features", "params", "TabM", "CatBoostClassifier", "roc_auc_score"]):
            print("CODE:", src[:400])
            print("---")
