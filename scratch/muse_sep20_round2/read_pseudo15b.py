"""Round2: pseudo15 acceptance outcome + fold seed + student OOF numbers."""
import json
import re

nb = json.load(open("scratch/muse_sep20_round2/nb_pseudo15/pseudo-labels-without-lying-to-cv-15-teachers.ipynb"))
full = "\n".join("".join(c.get("source", [])) for c in nb["cells"])
out = "\n".join(
    "".join(o.get("text", [])) for c in nb["cells"] for o in c.get("outputs", [])
    if o.get("output_type") == "stream")
print("== SEED ==")
for pat in [r"[Ss][Ee][Ee][Dd]\s*=\s*\d+", r"random_state\s*=\s*\d+",
            r"RANDOM[^\n]{0,40}", r"seed[^\n]{0,40}"]:
    print(pat, "->", sorted(set(re.findall(pat, full)))[:8])
print("== OUTCOME ==")
for pat in [r"accepted[^\n]{0,160}", r"delta[^\n]{0,120}", r"wins[^\n]{0,120}",
            r"OOF[^\n]{0,120}", r"candidate[^\n]{0,120}"]:
    print(pat, "->", sorted(set(re.findall(pat, out)))[:8])
