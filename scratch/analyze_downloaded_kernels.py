import os
import json

kernels_dir = "scratch/kernels"

for kd in sorted(os.listdir(kernels_dir)):
    full_path = os.path.join(kernels_dir, kd)
    if not os.path.isdir(full_path):
        continue
    print(f"\n=======================================================")
    print(f"KERNEL: {kd}")
    print(f"=======================================================")
    files = os.listdir(full_path)
    print("Files:", files)
    
    # Check kernel-metadata.json
    meta_file = os.path.join(full_path, "kernel-metadata.json")
    if os.path.exists(meta_file):
        with open(meta_file, 'r') as f:
            try:
                meta = json.load(f)
                print(f"Title: {meta.get('title')}")
                print(f"Competition: {meta.get('competition_sources')}")
                print(f"Dataset sources: {meta.get('dataset_sources')}")
                print(f"Kernel sources: {meta.get('kernel_sources')}")
            except Exception as e:
                print("Error loading meta:", e)

    # Check ipynb or py file
    for f in files:
        if f.endswith(".ipynb"):
            ip_path = os.path.join(full_path, f)
            with open(ip_path, 'r', encoding='utf-8', errors='ignore') as ipf:
                try:
                    nb = json.load(ipf)
                    cells = nb.get('cells', [])
                    print(f"Notebook cells: {len(cells)}")
                    # Find markdown cells with overview or conclusion
                    md_text = ""
                    for cell in cells[:5]:
                        if cell.get('cell_type') == 'markdown':
                            md_text += "".join(cell.get('source', [])) + "\n---\n"
                    print("Intro MD (up to 600 chars):")
                    print(md_text[:600])
                    
                    # Look for code lines with model definitions or CV/LB
                    code_lines = []
                    for cell in cells:
                        if cell.get('cell_type') == 'code':
                            src = "".join(cell.get('source', []))
                            for line in src.split("\n"):
                                if any(kw in line.lower() for kw in ['auc', 'score', 'catboost', 'realmlp', 'cv', 'oof', 'band', 'round(', 'rank']):
                                    if len(line.strip()) > 0 and not line.strip().startswith('#'):
                                        code_lines.append(line.strip())
                    print("Sample interesting code lines (first 10):")
                    for cl in code_lines[:10]:
                        print("  ", cl[:120])
                except Exception as e:
                    print("Error parsing ipynb:", e)
        elif f.endswith(".py"):
            py_path = os.path.join(full_path, f)
            with open(py_path, 'r', encoding='utf-8', errors='ignore') as pyf:
                content = pyf.read()
                print(f"Python script length: {len(content)} chars")
                print("Head (up to 500 chars):")
                print(content[:500])

