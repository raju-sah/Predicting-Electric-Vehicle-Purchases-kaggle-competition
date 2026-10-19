import subprocess

cmd = ["kaggle", "kernels", "list", "--competition", "playground-series-s6e9", "--sort-by", "dateRun", "--page-size", "20"]
res = subprocess.run(cmd, capture_output=True, text=True)
print(res.stdout)
