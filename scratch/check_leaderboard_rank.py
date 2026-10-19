import subprocess
import json

res = subprocess.run(["kaggle", "competitions", "leaderboard", "-c", "playground-series-s6e9", "--show"], capture_output=True, text=True)
lines = res.stdout.splitlines()

# Search for user or score 0.94649
print("Leaderboard slice around 0.94649:")
for line in lines:
    if "0.94649" in line or "0.94650" in line or "0.94651" in line or "0.94652" in line or "0.94653" in line or "0.94654" in line or "raju" in line.lower():
        print(line)
