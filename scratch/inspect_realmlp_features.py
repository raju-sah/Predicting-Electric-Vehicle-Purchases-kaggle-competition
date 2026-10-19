import json
import glob

files = glob.glob("scratch/**/ps-s6-e9-realmlp-pytorch*.ipynb", recursive=True)
print("RealMLP files found:", files)

if files:
    with open(files[0]) as f:
        nb = json.load(f)
    for i, cell in enumerate(nb['cells']):
        source = "".join(cell['source'])
        if "feature" in source.lower() or "preprocess" in source.lower() or "class " in source.lower():
            print(f"=== Cell {i} ===")
            print(source[:1000])
            print("\n" + "="*50 + "\n")
