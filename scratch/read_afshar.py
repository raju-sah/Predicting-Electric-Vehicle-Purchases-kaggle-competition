import json
nb = json.load(open("scratch/pull_sep22/afshar_spikes/s6e9-53-income-spikes-94-of-the-gain.ipynb"))
print("cells:", len(nb["cells"]))
for i, c in enumerate(nb["cells"]):
    src = "".join(c["source"])
    if c["cell_type"] == "markdown":
        print(f"--- MD {i} ---")
        print(src[:1500])
    else:
        print(f"--- CODE {i} ({len(src)} chars) ---")
        print(src[:1200])
    print()
