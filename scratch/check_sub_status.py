import subprocess, time

for i in range(6):
    res = subprocess.run(["kaggle", "competitions", "submissions", "-c", "playground-series-s6e9"], capture_output=True, text=True)
    lines = res.stdout.strip().split("\n")
    if len(lines) >= 3:
        header = lines[0]
        latest = lines[2]
        print(f"[{i*10}s] {latest}")
        if "COMPLETE" in latest:
            break
    time.sleep(10)
