import subprocess

cmd = ["kaggle", "kernels", "list", "--competition", "playground-series-s6e9", "--sort-by", "dateRun", "--page-size", "25"]
res = subprocess.run(cmd, capture_output=True, text=True)
print("Latest 25 kernels by dateRun:")
print(res.stdout)
