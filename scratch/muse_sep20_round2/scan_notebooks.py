"""Round2: extract claims (OOF/public/folds) from pulled Sep-20 notebooks."""
import json
import re

for tag, path in [
    ("embedmlp", "scratch/muse_sep20_round2/nb_embedmlp/pytorch-embedding-mlp-lightgbm-blend.ipynb"),
    ("pseudo15", "scratch/muse_sep20_round2/nb_pseudo15/pseudo-labels-without-lying-to-cv-15-teachers.ipynb"),
    ("devashish", "scratch/muse_sep20_round2/nb_devashish/predicting-electric-vehicle-purchases.ipynb"),
]:
    nb = json.load(open(path))
    texts = []
    for cell in nb.get("cells", []):
        src = "".join(cell.get("source", []))
        texts.append(f"--- {cell.get('cell_type')} ---\n{src[:1500]}")
        for out in cell.get("outputs", []):
            if out.get("output_type") == "stream":
                texts.append(">>> output: " + "".join(out.get("text", []))[:800])
            for d in out.get("data", {}).get("text/plain", []):
                texts.append(">>> data: " + d[:400])
    full = "\n".join(texts)
    print(f"===== {tag} ({len(nb.get('cells', []))} cells) =====")
    for pat in [r"0\.94\d{3}", r"AUC[^\n]{0,80}", r"auc[^\n]{0,80}",
                r"StratifiedKFold[^\n]{0,100}", r"n_splits[^\n]{0,60}",
                r"LB[^\n]{0,60}", r"public[^\n]{0,60}", r"fold[^\n]{0,60}"]:
        for m in sorted(set(re.findall(pat, full)))[:12]:
            print(f"  [{pat[:12]}] {m[:130]}")
    print()
