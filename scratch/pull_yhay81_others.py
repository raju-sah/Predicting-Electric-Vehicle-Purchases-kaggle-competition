import subprocess

for k in ["yhay81/simple-tabm-baseline-s6e9", "yhay81/simple-catboost-baseline-s6e9"]:
    cmd = ["kaggle", "kernels", "pull", "-p", "scratch/yhay81/", k]
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(k, res.stdout, res.stderr)
