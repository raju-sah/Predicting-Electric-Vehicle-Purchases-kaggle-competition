import subprocess
import json

cmd = ["kaggle", "competitions", "leaderboard", "-c", "playground-series-s6e9", "--show", "--format", "json"]
res = subprocess.run(cmd, capture_output=True, text=True)
try:
    data = json.loads(res.stdout)
    for rank, row in enumerate(data, 1):
        team = row.get("teamName", "")
        members = row.get("teamMembers", [])
        score = row.get("score", "")
        if "raju" in team.lower() or any("raju" in str(m).lower() for m in members):
            print(f"FOUND: Rank {rank} | Team: {team} | Score: {score}")
            # print surrounding 5 teams
            start = max(0, rank - 3)
            end = min(len(data), rank + 3)
            for r in range(start, end):
                print(f"  Rank {r+1}: {data[r].get('teamName')} - {data[r].get('score')}")
            break
except Exception as e:
    print(f"Error parsing JSON: {e}")
