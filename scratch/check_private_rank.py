"""Page through the private leaderboard and report our final rank."""
import re
import subprocess

CMP = "playground-series-s6e9"
token, seen, rows = None, set(), []
page = 0
while page < 10:
    cmd = ["kaggle", "competitions", "leaderboard", "-c", CMP, "--show", "--csv",
           "--page-size", "200"]
    if token:
        cmd += ["--page-token", token]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout
    m = re.search(r"Next Page Token = (\S+)", out)
    token = m.group(1) if m else None
    lines = [l for l in out.splitlines() if l and not l.startswith(("teamId", "Next Page"))]
    fresh = [l for l in lines if l not in seen]
    seen.update(fresh)
    rows.extend(fresh)
    page += 1
    if not token or not fresh:
        break

print(f"private leaderboard rows fetched: {len(rows)}")
for i, line in enumerate(rows, start=1):
    if "raju sah" in line.lower():
        cols = [c.strip() for c in line.split(",")]
        print(f"MATCH at row {i}: {cols}")
        break
else:
    print("no match found")