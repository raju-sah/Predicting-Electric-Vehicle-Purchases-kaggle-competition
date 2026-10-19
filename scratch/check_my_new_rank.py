import subprocess
import glob
import pandas as pd

subprocess.run(["kaggle", "competitions", "leaderboard", "-c", "playground-series-s6e9", "-d", "-p", "scratch/lb_new/"])
subprocess.run(["unzip", "-o", "scratch/lb_new/*.zip", "-d", "scratch/lb_new/"])

f = glob.glob("scratch/lb_new/*publicleaderboard*.csv")[0]
df = pd.read_csv(f).sort_values(by="Score", ascending=False).reset_index(drop=True)
df['Rank'] = df.index + 1

match = df[df['TeamName'].str.contains('raju', case=False, na=False) | df['TeamMemberUserNames'].str.contains('raju', case=False, na=False)]
for idx, row in match.iterrows():
    print(f"\n>>> Rank {row['Rank']}: Team '{row['TeamName']}' | Score: {row['Score']} | Submissions: {row['SubmissionCount']}")
    start = max(0, row['Rank'] - 5)
    end = min(len(df), row['Rank'] + 5)
    print(df.loc[start:end, ['Rank', 'TeamName', 'Score', 'SubmissionCount']].to_string())
