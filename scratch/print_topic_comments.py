import subprocess

res = subprocess.run(["kaggle", "competitions", "topics", "show", "-c", "playground-series-s6e9", "740769", "-v"], capture_output=True, text=True)
print("CSV format output:")
print(res.stdout[:4000])
