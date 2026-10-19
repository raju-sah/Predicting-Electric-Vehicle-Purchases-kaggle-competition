import subprocess

res = subprocess.run(["kaggle", "competitions", "topics", "-c", "playground-series-s6e9", "--page-size", "25"], capture_output=True, text=True)
print("Forum topics:")
print(res.stdout)
