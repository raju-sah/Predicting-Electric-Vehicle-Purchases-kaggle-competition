import json

path = "scratch/kernels/defiaudit_s6e9-rule-recipe-decoded/s6e9-rule-recipe-decoded.ipynb"
with open(path) as f:
    nb = json.load(f)

for i, c in enumerate(nb['cells']):
    if c['cell_type'] == 'markdown':
        src = "".join(c['source'])
        print(f"[MD {i}]:\n{src}\n---")
    elif c['cell_type'] == 'code':
        src = "".join(c['source'])
        print(f"[CODE {i}]:\n{src[:400]}\n---")
