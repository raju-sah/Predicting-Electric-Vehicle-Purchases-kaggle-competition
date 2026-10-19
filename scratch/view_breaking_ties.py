import json

for nb_name in ["chinzorigtganbat_s6e9-does-breaking-ties-help", "beraterolelk_0-94650-why-breaking-ties-wins-s6e9-guide"]:
    print(f"\n==================== {nb_name} ====================")
    path = f"scratch/kernels/{nb_name}/{nb_name.split('_', 1)[1]}.ipynb"
    try:
        with open(path) as f:
            nb = json.load(f)
        for c in nb['cells']:
            if c['cell_type'] == 'markdown':
                src = "".join(c['source'])
                if any(kw in src.lower() for kw in ['tie', 'lexsort', 'round', '0.946', 'rank']):
                    print("MD:", src[:500], "\n---")
            elif c['cell_type'] == 'code':
                src = "".join(c['source'])
                if any(kw in src.lower() for kw in ['lexsort', 'tie', 'round', 'rank']):
                    print("CODE:", src[:500], "\n---")
    except Exception as e:
        print("Error:", e)
