import subprocess

cmd = ["kaggle", "kernels", "list", "--competition", "playground-series-s6e9", "--search", "0.946", "--page-size", "20"]
res = subprocess.run(cmd, capture_output=True, text=True)
print("0.946 kernels:")
print(res.stdout)
