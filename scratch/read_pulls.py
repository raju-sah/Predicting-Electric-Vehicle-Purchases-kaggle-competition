import json
for path in ["scratch/pull_sep22/busya_gpu/pure-lgbm-cv-0-94607-lb-0-94638-3-gpu-models.ipynb",
             "scratch/pull_sep22/jev_te/s6e9-jev-multiscale-te-v1.ipynb"]:
    nb = json.load(open(path))
    print("=" * 80)
    print(path, "cells:", len(nb["cells"]))
    for i, c in enumerate(nb["cells"]):
        src = "".join(c["source"])
        tag = "MD" if c["cell_type"] == "markdown" else "CODE"
        print(f"--- {tag} {i} ({len(src)} chars) ---")
        print(src[:1000])
    print()
