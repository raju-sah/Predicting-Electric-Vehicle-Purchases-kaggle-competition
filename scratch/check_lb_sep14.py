import subprocess

res = subprocess.run(["kaggle", "competitions", "leaderboard", "-c", "playground-series-s6e9", "--show"], capture_output=True, text=True)
lines = res.stdout.splitlines()

print("Current Leaderboard Top 35 on Sep 14:")
for i, line in enumerate(lines[:38]):
    print(f"{i:2d}: {line}")

print("\nFinding Raju Sah:")
for i, line in enumerate(lines[2:]):
    if "raju" in line.lower():
        print(f"Rank {i+1}: {line}")
