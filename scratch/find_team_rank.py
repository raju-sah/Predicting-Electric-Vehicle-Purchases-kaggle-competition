import subprocess

res = subprocess.run(["kaggle", "competitions", "leaderboard", "-c", "playground-series-s6e9", "--show", "-p", "2"], capture_output=True, text=True)
print("Page 2 slice:")
for line in res.stdout.splitlines()[:25]:
    print(line)

res1 = subprocess.run(["kaggle", "competitions", "leaderboard", "-c", "playground-series-s6e9", "--show"], capture_output=True, text=True)
for i, line in enumerate(res1.stdout.splitlines()):
    if "16843656" in line or "raju" in line.lower():
        print(f"Found on page 1: {line}")
for i, line in enumerate(res.stdout.splitlines()):
    if "16843656" in line or "raju" in line.lower():
        print(f"Found on page 2: {line}")
