import json

with open("scratch/kernels/najiama_catboost-triple-te-dynamic-pruning-lb-0-94624/catboost-triple-te-dynamic-pruning-lb-0-94624.ipynb") as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'code':
        src = "".join(c['source'])
        if "CatBoostClassifier" in src or "clf" in src or "roc_auc_score" in src:
            print(f"--- Cell {i} ---")
            print(src[:1200])
