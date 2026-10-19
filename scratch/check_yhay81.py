import subprocess

cmd = ["kaggle", "kernels", "list", "--user", "yhay81"]
res = subprocess.run(cmd, capture_output=True, text=True)
print("yhay81 kernels:")
print(res.stdout)
print(res.stderr)

cmd2 = ["kaggle", "kernels", "pull", "-p", "scratch/yhay81/", "yhay81/exact-reconstruction-of-the-ev-source-dataset"]
res2 = subprocess.run(cmd2, capture_output=True, text=True)
print("Pull reconstruction result:", res2.stdout, res2.stderr)
