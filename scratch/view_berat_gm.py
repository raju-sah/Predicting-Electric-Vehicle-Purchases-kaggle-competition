import json

path = "scratch/kernels/beraterolelk_lb-0-94656-top-38-ev-purchase-gm-ensemble/lb-0-94656-top-38-ev-purchase-gm-ensemble.ipynb"
with open(path) as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'markdown':
        src = "".join(c['source'])
        print(f"[MD {i}]:\n{src[:400]}\n---")
    elif c['cell_type'] == 'code':
        src = "".join(c['source'])
        if any(kw in src.lower() for kw in ["0.946", "blend", "rank", "weights"]):
            print(f"[CODE {i}]:\n{src[:400]}\n---")
