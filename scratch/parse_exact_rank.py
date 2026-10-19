import glob
import pandas as pd

f = glob.glob("scratch/lb_dl/*publicleaderboard*.csv")[0]
df = pd.read_csv(f)
print(f"Total teams on leaderboard: {len(df)}")

df = df.sort_values(by="Score", ascending=False).reset_index(drop=True)
df['Rank'] = df.index + 1

match = df[df['TeamName'].str.contains('raju', case=False, na=False) | df['TeamMemberUserNames'].str.contains('raju', case=False, na=False)]
if len(match) == 0:
    print("Top 50 teams:")
    print(df[['Rank', 'TeamName', 'Score', 'SubmissionCount']].head(50).to_string())
else:
    print("Found team matches:")
    for idx, row in match.iterrows():
        print(f"\n>>> Rank {row['Rank']}: Team '{row['TeamName']}' | Score: {row['Score']} | Submissions: {row['SubmissionCount']} | Users: {row['TeamMemberUserNames']}")
        start = max(0, row['Rank'] - 6)
        end = min(len(df), row['Rank'] + 6)
        print(df.loc[start:end, ['Rank', 'TeamName', 'Score', 'SubmissionCount']].to_string())
