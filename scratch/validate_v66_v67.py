import pandas as pd
for f in ["subs/v66_verified5_probability.csv", "subs/v67_verified5_rank.csv", "sample_submission.csv"]:
    df = pd.read_csv(f)
    print(f, df.shape, list(df.columns), df.iloc[0].to_dict(), df.iloc[-1].to_dict())
s = pd.read_csv("sample_submission.csv")["id"]
for f in ["subs/v66_verified5_probability.csv", "subs/v67_verified5_rank.csv"]:
    sub = pd.read_csv(f)
    ok_ids = (sub["id"].values == s.values).all()
    ok_range = sub.iloc[:, 1].between(0, 1).all()
    print(f, "ids_match=", ok_ids, "probs_ok=", bool(ok_range), "nulls=", int(sub.isnull().sum().sum()))
