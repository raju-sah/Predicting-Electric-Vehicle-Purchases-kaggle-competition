import subprocess
res = subprocess.run(["kaggle", "competitions", "topic-messages", "--help"], capture_output=True, text=True)
print(res.stdout)
print(res.stderr)
