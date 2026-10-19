import subprocess

res = subprocess.run(["kaggle", "kernels", "list", "-s", "s6e9", "--sort-by", "dateRun", "--page-size", "30"], capture_output=True, text=True)
print("Latest s6e9 kernels:")
print(res.stdout)
