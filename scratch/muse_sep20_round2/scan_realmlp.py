"""Round2: yekenot RealMLP notebook essence for escalation path."""
import json
import re

nb = json.load(open("scratch/realmlp/ps-s6-e9-realmlp-pytorch.ipynb"))
print("cells:", len(nb.get("cells", [])))
full = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    if cell.get("cell_type") == "markdown" and len(src.strip()) > 60:
        print(f"--- MD {i} ---")
        print(src[:700])
        print()
for pat in [r"0\.94\d+", r"hidden[^\n]{0,60}", r"epoch[^\n]{0,60}",
            r"lr[^\n]{0,40}", r"batch[^\n]{0,40}", r"StratifiedKFold[^\n]{0,60}",
            r"embedding[^\n]{0,80}"]:
    print(pat, "->", sorted(set(re.findall(pat, full)))[:8])
