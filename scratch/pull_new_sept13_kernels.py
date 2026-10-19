import subprocess

kernels = [
    "dariushafshar/s6e9-0-94585-1-of-4-ingredients-carries-it",
    "najiama/xgboost-triple-te-dynamic-pruning-lb-0-94639",
    "taeyangg4/s6e9-094645-multi-paradigm-stacking",
    "amirhosseinkarimiee/the-noise-bar-feature-engineering"
]

for k in kernels:
    user, slug = k.split("/")
    dest = f"scratch/{user}/"
    cmd = ["kaggle", "kernels", "pull", "-p", dest, k]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(k, res.stdout, res.stderr)
    
    cmd_out = ["kaggle", "kernels", "output", k, "-p", f"scratch/{user}_out/"]
    res_out = subprocess.run(cmd_out, capture_output=True, text=True)
    print(f"{k} output:", res_out.stdout[:300], res_out.stderr[:300])
