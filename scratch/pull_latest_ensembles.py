import subprocess

kernels = [
    "tamerlanomralinov/s6e9-hirge-net-and-doubly-nested-maxcv-stack",
    "nina2025/ps-s6e9-honest-round"
]

for k in kernels:
    user, slug = k.split("/")
    dest = f"scratch/{user}/"
    cmd = ["kaggle", "kernels", "pull", "-p", dest, k]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(k, res.stdout, res.stderr)
    
    cmd_out = ["kaggle", "kernels", "output", k, "-p", f"scratch/{user}_out/"]
    res_out = subprocess.run(cmd_out, capture_output=True, text=True)
    print(f"{k} output:", res_out.stdout, res_out.stderr)
