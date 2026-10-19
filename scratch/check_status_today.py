import subprocess
import json

cmd = ["kaggle", "competitions", "submissions", "-c", "playground-series-s6e9"]
res = subprocess.run(cmd, capture_output=True, text=True)
print("Submissions history:")
print(res.stdout[:1500])

cmd_lb = ["kaggle", "competitions", "leaderboard", "-c", "playground-series-s6e9", "--show"]
res_lb = subprocess.run(cmd_lb, capture_output=True, text=True)
print("\nLeaderboard head:")
print(res_lb.stdout[:1500])
