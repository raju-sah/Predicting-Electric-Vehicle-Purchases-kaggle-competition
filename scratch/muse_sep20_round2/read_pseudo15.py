"""Round2: deep read of pseudo15 notebook narrative + key params."""
import json
import re

nb = json.load(open("scratch/muse_sep20_round2/nb_pseudo15/pseudo-labels-without-lying-to-cv-15-teachers.ipynb"))
for i, cell in enumerate(nb.get("cells", [])):
    src = "".join(cell.get("source", []))
    if cell.get("cell_type") == "markdown":
        print(f"##### MD cell {i} #####")
        print(src[:2000])
        print()
    outs = []
    for out in cell.get("outputs", []):
        if out.get("output_type") == "stream":
            outs.append("".join(out.get("text", []))[:600])
    if outs:
        print(f"##### CODE cell {i} outputs #####")
        print("\n".join(outs)[:1500])
        print()
# key params
full = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
for pat in [r"SEED\s*=\s*\d+", r"N_FOLDS\s*=\s*\d+", r"teacher[^\n]{0,100}",
            r"0\.946\d+", r"accepted[^\n]{0,120}", r"pseudo[^\n]{0,120}"]:
    hits = sorted(set(re.findall(pat, full)))[:10]
    print(pat, "->", hits)
