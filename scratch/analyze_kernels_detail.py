import os
import json

targets = [
    "beraterolelk_lb-0-94656-top-38-ev-purchase-gm-ensemble",
    "chinzorigtganbat_s6e9-does-breaking-ties-help",
    "beraterolelk_0-94650-why-breaking-ties-wins-s6e9-guide",
    "beraterolelk_0-9385-oof-ev-pytorch-tabular-stacker",
    "defiaudit_s6e9-rule-mining-at-lb-resolution",
    "kiza123123_s6e9-generator-forensics-5-submission-pipeline"
]

for t in targets:
    p = os.path.join("scratch/kernels", t)
    if not os.path.exists(p):
        continue
    print(f"\n=======================================================")
    print(f"DETAILS: {t}")
    print(f"=======================================================")
    for fname in os.listdir(p):
        if fname.endswith(".ipynb"):
            with open(os.path.join(p, fname), 'r', encoding='utf-8', errors='ignore') as f:
                nb = json.load(f)
                for i, cell in enumerate(nb.get("cells", [])):
                    if cell.get("cell_type") == "markdown":
                        text = "".join(cell.get("source", []))
                        if any(kw in text.lower() for kw in ["0.946", "tie", "trick", "method", "private", "shakeup", "architecture", "blend", "model"]):
                            print(f"[MD cell {i}]:\n{text[:500]}\n---")
                    elif cell.get("cell_type") == "code":
                        text = "".join(cell.get("source", []))
                        if any(kw in text.lower() for kw in ["round", "lexsort", "rank", "nn", "pytorch", "catboost", "0.946"]):
                            print(f"[Code cell {i}]:\n{text[:400]}\n---")
        elif fname.endswith(".py"):
            with open(os.path.join(p, fname), 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                print(f"[Python file {fname}, total lines {len(lines)}]")
                print("First 30 lines:\n" + "".join(lines[:30]))

