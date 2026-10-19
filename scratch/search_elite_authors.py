import subprocess

authors = ["cdeotte", "yhay81", "donmani", "georgymamarin", "tilii7"]

for author in authors:
    print(f"\n==================== AUTHOR: {author} ====================")
    res_k = subprocess.run(["kaggle", "kernels", "list", "-u", author, "-s", "s6e9"], capture_output=True, text=True)
    print("KERNELS:")
    print(res_k.stdout if res_k.returncode == 0 else res_k.stderr)
    
    res_d = subprocess.run(["kaggle", "datasets", "list", "-u", author, "-s", "s6e9"], capture_output=True, text=True)
    print("DATASETS:")
    print(res_d.stdout if res_d.returncode == 0 else res_d.stderr)
