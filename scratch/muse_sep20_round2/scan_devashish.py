"""Round2: devashish notebook essence — is it a band-swap copy or new?"""
import json
import re

nb = json.load(open("scratch/muse_sep20_round2/nb_devashish/predicting-electric-vehicle-purchases.ipynb"))
print("cells:", len(nb.get("cells", [])))
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    if cell.get("cell_type") == "markdown" and len(src.strip()) > 40:
        print(f"--- MD {i} ---")
        print(src[:900])
        print()
        if i > 12:
            break
full = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
for pat in [r"0\.946\d+", r"band[^\n]{0,80}", r"swap[^\n]{0,80}",
            r"LB[^\n]{0,80}", r"StratifiedKFold[^\n]{0,80}"]:
    print(pat, "->", sorted(set(re.findall(pat, full)))[:8])
