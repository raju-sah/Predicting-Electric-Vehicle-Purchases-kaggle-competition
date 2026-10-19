import subprocess

cmd = ["kaggle", "kernels", "pull", "-p", "scratch/cdeotte/", "cdeotte/fable-5-1-xgb-starter"]
res = subprocess.run(cmd, capture_output=True, text=True)
print(res.stdout, res.stderr)
