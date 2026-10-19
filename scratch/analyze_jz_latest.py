import numpy as np, pandas as pd, hashlib, json

cat = pd.read_csv("scratch/pull_sep16/jz_meta/ranked_catalog_latest.csv")
print("catalog cols:", list(cat.columns)[:14], "... rows:", len(cat))

best = pd.read_csv("scratch/pull_sep16/jz_meta/submission_latest_best.csv")
h = hashlib.sha256(open("scratch/pull_sep16/jz_meta/submission_latest_best.csv","rb").read()).hexdigest()
print("sha256:", h[:16], "| catalog best sha:", cat.iloc[0].get("sha256","?")[:16] if len(cat) else "?")

sample = pd.read_csv("sample_submission.csv")
ok = (best["id"].values == sample["id"].values).all()
print("ids match sample:", ok, "| rows:", len(best), "| header:", list(best.columns))
print("prob range:", best.iloc[:,1].min(), best.iloc[:,1].max(), "| ties(2dp):", (best.iloc[:,1].round(2).duplicated()).sum())

anchor = pd.read_csv("subs/v36_apex_upgraded_slots.csv")["Will_Buy_EV"].values
b = best["Will_Buy_EV"].values
rb, ra = pd.Series(b).rank(method="average").values, pd.Series(anchor).rank(method="average").values
from scipy.stats import spearmanr
print("spearman vs our v36 apex: %.6f" % spearmanr(b, anchor).statistic)

# top catalog entries
sc_col = [c for c in cat.columns if "score" in c.lower() and "display" not in c.lower()]
print("\nTOP 15 catalog rows:")
cols = [c for c in cat.columns if c.lower() in ("display_order","displayed_score","public_score","file_path","source_url","author","displayed_score_rank","file_name")]
print(cat[cols].head(15).to_string())

# how many distinct top files share the best displayed score
top_score = cat[cols[1]].iloc[0] if len(cols)>1 else None
n_top = (cat[cols[1]] == top_score).sum() if top_score is not None else "?"
print("\nfiles sharing best displayed score:", n_top)
