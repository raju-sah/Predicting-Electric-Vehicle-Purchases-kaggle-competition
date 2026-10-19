import subprocess

print("--- Checking kaggle CLI forum/competition commands ---")
res = subprocess.run(["kaggle", "competitions", "--help"], capture_output=True, text=True)
print(res.stdout)
