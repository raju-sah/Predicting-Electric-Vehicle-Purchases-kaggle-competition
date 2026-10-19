import glob
import pandas as pd

f = glob.glob("scratch/lb_dl/*publicleaderboard*.csv")[0]
df = pd.read_csv(f).sort_values(by="Score", ascending=False).reset_index(drop=True)
df['Rank'] = df.index + 1

print("Top 35 Leaderboard Standing:")
print(df.loc[0:34, ['Rank', 'TeamName', 'Score', 'SubmissionCount']].to_string())
