import numpy as np, pandas as pd, zipfile, hashlib, io
from scipy.stats import rankdata, spearmanr

TARGET, ID = "Will_Buy_EV", "id"

cat = pd.read_csv("scratch/pull_sep16/jz_meta/ranked_catalog_latest.csv")
outer = zipfile.ZipFile("scratch/pull_sep16/jz_full/s6e9-zoom-zoom-baseline.zip")
inner_bytes = outer.read("ranked_predictions_latest.zip.bin")
inner = zipfile.ZipFile(io.BytesIO(inner_bytes))
inner_names = set(inner.namelist())
print("inner entries:", len(inner_names))

def sha(b): return hashlib.sha256(b).hexdigest()

# find all 0.94651 entries + top few 0.94650 for reference
top51 = cat[cat["public_score"] == 0.94651]
top50 = cat[cat["public_score"] == 0.94650].head(8)
print("\n0.94651 entries:")
print(top51[["author", "submission_id", "version_id", "sha256", "csv_path", "prediction_kind"]].to_string())

sample = pd.read_csv("sample_submission.csv")
ids = sample[ID].to_numpy()

def pull(sha_prefix):
    rows = cat[cat["sha256"].str.startswith(sha_prefix)]
    assert len(rows) == 1, f"ambiguous sha {sha_prefix}: {len(rows)}"
    row = rows.iloc[0]
    # find the file inside inner zip by matching hash of contents
    for n in inner_names:
        if not n.endswith(".csv"): continue
        b = inner.read(n)
        if sha(b).startswith(sha_prefix):
            d = pd.read_csv(io.BytesIO(b)).sort_values(ID)
            assert np.array_equal(d[ID].to_numpy(), ids)
            return d[TARGET].to_numpy(dtype=float), row
    return None, row

# v46: rank-average of ALL distinct 0.94651 files
preds, metas = [], []
for h in top51["sha256"].unique():
    p, row = pull(h[:16])
    if p is not None:
        preds.append(p); metas.append(row["csv_path"])
        print("got:", row["csv_path"], "| sha", h[:12])
    else:
        print("MISSING from archive:", row["csv_path"], "| sha", h[:12])

print("\ncollected 0.94651 files:", len(preds), metas)
blend = np.mean([rankdata(p) / len(p) for p in preds], axis=0)
rho = spearmanr(blend, preds[0]).statistic
print(f"blend rho vs first 0.94651 file: {rho:.6f}")

sub = pd.DataFrame({ID: ids, TARGET: blend})
assert sub[TARGET].nunique() == len(sub)
assert not sub[TARGET].isnull().any()
sub.to_csv("subs/v46_jz_trio_rankavg.csv", index=False)
print("SUCCESS subs/v46_jz_trio_rankavg.csv")

# also stash the other distinct 0.94651 files individually for future use
seen = {sha(open("scratch/pull_sep16/jz_meta/submission_latest_best.csv","rb").read())[:16]}
for h in top51["sha256"].unique():
    if h[:16] in seen: continue
    p, row = pull(h[:16])
    if p is not None:
        out = "scratch/pull_sep16/jz_meta/extra_094651_" + h[:12] + ".csv"
        pd.DataFrame({ID: ids, TARGET: p}).to_csv(out, index=False)
        print("stashed:", out)
