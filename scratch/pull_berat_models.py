import subprocess
import os

kernels = [
    "beraterolelk/s6e9-triblend-oof-ensemble-xgb-lgbm-catboost",
    "beraterolelk/s6e9-anchor-3seed-lgbm",
    "beraterolelk/s6e9-rebuilt-anchor-lgbm-original-data-3seed"
]

for k in kernels:
    name = k.replace("/", "_")
    target = f"scratch/kernels/{name}"
    os.makedirs(target, exist_ok=True)
    res = subprocess.run(f"kaggle kernels pull {k} -p {target} -m", shell=True, capture_output=True, text=True)
    print(f"Pulled {k}: returncode {res.returncode}")

