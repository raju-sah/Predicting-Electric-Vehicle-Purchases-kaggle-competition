import numpy as np, pandas as pd, zipfile, hashlib, io
from scipy.stats import rankdata, spearmanr

TARGET, ID = "Will_Buy_EV", "id"
cat = pd.read_csv("scratch/pull_sep16/jz_meta/ranked_catalog_latest.csv")

# 0) Chinzo re-publication check
h_chinzo = hashlib.sha256(open("scratch/pull_sep17/chinzo_ties/submission.csv","rb").read()).hexdigest()
h_base = hashlib.sha256(open("scratch/pull_sep16/jz_meta/submission_latest_best.csv","rb").read()).hexdigest()
print("chinzo sha:", h_chinzo[:16], "| base sha:", h_base[:16], "| identical:", h_chinzo == h_base)

# 1) pull all files with displayed score >= 0.94650
tier = cat[cat["public_score"] >= 0.94650].copy()
print("\ntier rows:", len(tier), "| distinct shas:", tier["sha256"].nunique())
print(tier.groupby(["public_score","origin"]).size())

outer = zipfile.ZipFile("scratch/pull_sep16/jz_full/s6e9-zoom-zoom-baseline.zip")
inner = zipfile.ZipFile(io.BytesIO(outer.read("ranked_predictions_latest.zip.bin")))
inner_names = [n for n in inner.namelist() if n.endswith(".csv")]
hash_map = {}
for n in inner_names:
    b = inner.read(n)
    hash_map[hashlib.sha256(b).hexdigest()] = (n, b)
print("\ninner csvs hashed:", len(hash_map))

sample = pd.read_csv("sample_submission.csv")
ids = sample[ID].to_numpy()

got = {}
for _, row in tier.iterrows():
    h = row["sha256"]
    if h in hash_map and h not in got:
        n, b = hash_map[h]
        d = pd.read_csv(io.BytesIO(b)).sort_values(ID)
        assert np.array_equal(d[ID].to_numpy(), ids), n
        got[h] = (d[TARGET].to_numpy(dtype=float), row["origin"], row["public_score"], n)
print("extracted distinct:", len(got))

# 2) rank matrix + dedupe identical-rank groups
M = {}
base_pred = got[h_base][0]
rb = rankdata(base_pred)
groups = []  # (rep_key, [members])
for h, (p, origin, sc, n) in got.items():
    r = rankdata(p)
    placed = False
    for g in groups:
        if spearmanr(r, rankdata(got[g[0]][0])).statistic > 0.999999:
            g.append(h); placed = True; break
    if not placed:
        groups.append([h])
print("\ndistinct rank groups:", len(groups))
for i, g in enumerate(groups):
    _, origin, sc, n = got[g[0]]
    rho = spearmanr(rankdata(got[g[0]][0]), rb).statistic
    print(f"  G{i}: n={len(g)} score={sc} origin={origin} rho_vs_base={rho:.6f} e.g. {n}")

# 3) save the distinct group reps
import json
meta = []
for i, g in enumerate(groups):
    h = g[0]
    p, origin, sc, n = got[h]
    out = f"scratch/pull_sep17/tier_G{i}.csv"
    pd.DataFrame({ID: ids, TARGET: p}).to_csv(out, index=False)
    meta.append({"group": i, "members": len(g), "score": float(sc), "origin": origin,
                 "file": n, "sha": h, "path": out})
json.dump(meta, open("scratch/pull_sep17/tier_groups.json","w"), indent=1)
print("\nsaved group reps -> scratch/pull_sep17/tier_G*.csv")
