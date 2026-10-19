import json

path = "scratch/kernels/denpugovkin_when-an-oof-gain-does-not-transfer-0-94649-guard/when-an-oof-gain-does-not-transfer-0-94649-guard.ipynb"
with open(path) as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'markdown':
        src = "".join(c['source'])
        print(f"[MD {i}]:\n{src}\n---")
    elif c['cell_type'] == 'code':
        src = "".join(c['source'])
        if any(kw in src.lower() for kw in ["0.946", "oof", "public", "transfer", "guard", "private", "shakeup"]):
            print(f"[CODE {i}]:\n{src[:400]}\n---")
