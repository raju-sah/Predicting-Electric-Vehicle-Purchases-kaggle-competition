import json

with open("scratch/kernels/najiama_catboost-triple-te-dynamic-pruning-lb-0-94624/catboost-triple-te-dynamic-pruning-lb-0-94624.ipynb") as f:
    nb = json.load(f)

src = "".join(nb['cells'][9]['source'])
idx = src.find("CatBoostClassifier")
print(src[idx:idx+1500])
