import subprocess
import json

cmd = ["kaggle", "competitions", "topics", "list", "-c", "playground-series-s6e9", "--page-size", "25"]
res = subprocess.run(cmd, capture_output=True, text=True)
print(res.stdout)
if res.stderr:
    print("STDERR:", res.stderr)
