import subprocess

res = subprocess.run(["kaggle", "competitions", "leaderboard", "-c", "playground-series-s6e9", "--show"], capture_output=True, text=True)
lines = res.stdout.splitlines()

rank = 0
for line in lines[2:]:
    parts = line.split()
    if len(parts) >= 2:
        rank += 1
        if "raju" in line.lower():
            print(f"EXACT RANK: {rank}")
            print(line)
            break
