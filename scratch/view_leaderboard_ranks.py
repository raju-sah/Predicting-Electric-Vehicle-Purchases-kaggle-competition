import subprocess
import pandas as pd
import io

res = subprocess.run(["kaggle", "competitions", "leaderboard", "-c", "playground-series-s6e9", "--show", "--csv"], capture_output=True, text=True)
if res.returncode == 0:
    df = pd.read_csv(io.StringIO(res.stdout))
    print("Columns:", df.columns.tolist())
    print("\nFirst 30 rows:")
    for idx, row in df.head(30).iterrows():
        print(dict(row))
else:
    print("Error:", res.stderr)
