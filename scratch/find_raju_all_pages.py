import subprocess

# Let's inspect pages using next-page-token
token = ""
found = False
rank = 0

for page in range(1, 5):
    cmd = ["kaggle", "competitions", "leaderboard", "-c", "playground-series-s6e9", "--show"]
    if token:
        cmd.extend(["--page-token", token])
    res = subprocess.run(cmd, capture_output=True, text=True)
    lines = res.stdout.splitlines()
    
    # Check next page token
    for line in lines:
        if "Next Page Token =" in line:
            token = line.split("=")[1].strip()
    
    for line in lines:
        if "teamId" in line or "---" in line or "Next Page" in line or not line.strip():
            continue
        rank += 1
        if "raju" in line.lower() or "16843656" in line:
            print(f"FOUND at Rank {rank}:")
            print(line)
            found = True
            break
    if found:
        break
