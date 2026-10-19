# v50-v52: last untested direction — tiny out-of-family injections on the 0.94651 base.
# Family bagging (v47-v49) read 0.94650; cross-family files (megayak stack 0.94645,
# nina 0.94645) are from different modeling lineages. If even 2% of decorrelated
# signal helps, LB should read 0.94652; if it hurts like family drift did, we learn
# the base is universally locally optimal. Span: 2%, 5%, and a 30% direct cross-mix.
import numpy as np, pandas as pd, zipfile, hashlib, io
from scipy.stats import rankdata, spearmanr

TARGET, ID = "Will_Buy_EV", "id"
sample = pd.read_csv("sample_submission.csv")
ids = sample[ID].to_numpy()
N = len(ids)

def lexrank(primary, secondary):
    o = np.lexsort((secondary, primary))
    r = np.empty(len(o), dtype=np.int64)
    r[o] = np.arange(1, len(o) + 1)
    return (r - 0.5) / len(r)

# pull out-of-family files from the nested archive by sha
cat = pd.read_csv("scratch/pull_sep16/jz_meta/ranked_catalog_latest.csv")
outer = zipfile.ZipFile("scratch/pull_sep16/jz_full/s6e9-zoom-zoom-baseline.zip")
inner = zipfile.ZipFile(io.BytesIO(outer.read("ranked_predictions_latest.zip.bin")))
hmap = {}
for n in inner.namelist():
    if n.endswith(".csv"):
        b = inner.read(n)
        hmap[hashlib.sha256(b).hexdigest()] = b

def pull(sha):
    d = pd.read_csv(io.BytesIO(hmap[sha])).sort_values(ID)
    assert np.array_equal(d[ID].to_numpy(), ids)
    return d[TARGET].to_numpy(dtype=float)

mg_stack = pull("adccb408b861e96daea33dc22aafeaf20a0d0223065d7cb37c0359316a60f78c")  # 0.94645 megayak/crystalbaby
nina    = pull("3aaf86cdf441ed3cf9673a7c45909b24c46dbb3cc76f2b9ed87f86b0a8fa88ac")   # 0.94645 nina2025

base = pd.read_csv("scratch/pull_sep16/jz_meta/submission_latest_best.csv").sort_values(ID)
base = base[TARGET].to_numpy(dtype=float)
rb = rankdata(base)
nn = pd.read_csv("scratch/pull_sep16/ds_megayak6/test_realmlp_g.csv").sort_values(ID)
nn = nn["G_realmlp_3seed"].to_numpy(dtype=float)

for h, nm in [(mg_stack, "mg"), (nina, "nina")]:
    r = rankdata(h)
    print(nm, "rho vs base:", round(spearmanr(r, rb).statistic, 6))

def emit(final, name, note):
    sub = pd.DataFrame({ID: ids, TARGET: final})
    assert sub[TARGET].nunique() == N and not sub[TARGET].isnull().any()
    assert np.array_equal(sub[ID].to_numpy(), sample[ID].to_numpy())
    sub.to_csv(f"subs/{name}.csv", index=False)
    print(f"SUCCESS subs/{name}.csv | {note} | rho vs base {spearmanr(final, base).statistic:.6f}")

r_mg, r_ni = rankdata(mg_stack) / N, rankdata(nina) / N
# v50: 2% out-of-family (megayak stack)
emit(lexrank(0.98 * rb / N + 0.02 * r_mg, nn), "v50_base_plus2pct_mgstack", "base + 2% megayak 0.94645 stack")
# v51: 5% out-of-family envelope
emit(lexrank(0.95 * rb / N + 0.05 * r_mg, nn), "v51_base_plus5pct_mgstack", "base + 5% megayak 0.94645 stack")
# v52: 30% cross-family mix (70% base + 15% mg + 15% nina)
emit(lexrank(0.70 * rb / N + 0.15 * r_mg + 0.15 * r_ni, nn), "v52_base_cross30", "70% base + 15% mg-stack + 15% nina")
