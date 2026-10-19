import subprocess
import os

kernels = [
    "najiama/catboost-triple-te-dynamic-pruning-lb-0-94624",
    "chinzorigtganbat/s6e9-does-breaking-ties-help",
    "beraterolelk/0-94650-why-breaking-ties-wins-s6e9-guide",
    "beraterolelk/lb-0-94656-top-38-ev-purchase-gm-ensemble",
    "beraterolelk/0-9385-oof-ev-pytorch-tabular-stacker",
    "denpugovkin/when-an-oof-gain-does-not-transfer-0-94649-guard",
    "defiaudit/s6e9-rule-mining-at-lb-resolution",
    "defiaudit/s6e9-rule-recipe-decoded",
    "kiza123123/s6e9-generator-forensics-5-submission-pipeline",
    "ksmashhero/s6e9-live-leaderboard-explorer-meta-analysis",
    "nina2025/ps-s6e9-top-subm-will-buy-ev-round-5"
]

os.makedirs("scratch/kernels", exist_ok=True)

for k in kernels:
    name = k.replace("/", "_")
    target_dir = f"scratch/kernels/{name}"
    os.makedirs(target_dir, exist_ok=True)
    print(f"Pulling {k}...")
    res = subprocess.run(f"kaggle kernels pull {k} -p {target_dir} -m", shell=True, capture_output=True, text=True)
    if res.returncode == 0:
        print(f"  Success: {name}")
    else:
        print(f"  Failed: {res.stderr.strip()}")

