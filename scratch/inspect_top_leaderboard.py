import subprocess

res = subprocess.run(["kaggle", "competitions", "leaderboard", "-c", "playground-series-s6e9", "--show"], capture_output=True, text=True)
lines = res.stdout.splitlines()

print("Top 25 on Leaderboard:")
for i, line in enumerate(lines[:28]):
    print(f"{i:2d}: {line}")
