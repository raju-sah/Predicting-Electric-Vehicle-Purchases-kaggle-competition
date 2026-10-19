import json

with open("scratch/kernels_pulled/taeyangg4_playbook/s6e9-0-94644-lb-vs-oof-slot-saving-playbook.ipynb") as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    source = "".join(cell['source'])
    if cell['cell_type'] == 'markdown' or any(k in source.lower() for k in ["blend", "weight", "stack", "regime", "exp055"]):
        print(f"=== Cell {i} [{cell['cell_type']}] ===")
        print(source[:1200])
        print("\n" + "="*50 + "\n")
