import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

TARGET, ID = "Will_Buy_EV", "id"
sample = pd.read_csv("sample_submission.csv")
ids = sample[ID].to_numpy()
n = len(ids)


def rank01(values):
    return rankdata(values, method="average") / n


def lexrank(primary, secondary):
    order = np.lexsort((secondary, primary))
    ranks = np.empty(len(order), dtype=np.int64)
    ranks[order] = np.arange(1, len(order) + 1)
    return (ranks - 0.5) / len(ranks)


base_df = pd.read_csv("scratch/pull_sep16/jz_meta/submission_latest_best.csv").sort_values(ID)
model_df = pd.read_csv("scratch/pull_sep18/memory_recipe_out/submission_model_only.csv").sort_values(ID)
secondary_df = pd.read_csv("scratch/pull_sep16/ds_megayak6/test_realmlp_g.csv").sort_values(ID)
for frame in (base_df, model_df, secondary_df):
    assert np.array_equal(frame[ID].to_numpy(), ids)

base = base_df[TARGET].to_numpy(float)
model = model_df[TARGET].to_numpy(float)
secondary = secondary_df["G_realmlp_3seed"].to_numpy(float)
r_base, r_model = rank01(base), rank01(model)
print(f"model rho vs 0.94651 base: {spearmanr(r_model, r_base).statistic:.6f}")

for version, weight in (("v53", 0.02), ("v54", 0.05), ("v55", 0.10), ("v56", 0.15)):
    final = lexrank((1.0 - weight) * r_base + weight * r_model, secondary)
    out = pd.DataFrame({ID: ids, TARGET: final})
    assert out.columns.tolist() == [ID, TARGET]
    assert len(out) == len(sample)
    assert np.array_equal(out[ID].to_numpy(), ids)
    assert np.isfinite(out[TARGET]).all()
    assert out[TARGET].between(0, 1).all()
    assert out[TARGET].nunique() == n
    path = f"subs/{version}_jz_memory_residual_w{int(weight * 100):02d}.csv"
    out.to_csv(path, index=False)
    rho = spearmanr(final, base).statistic
    displacement = np.abs(rankdata(final) - rankdata(base))
    print(
        f"{path}: weight={weight:.2f}, rho={rho:.9f}, "
        f"mean_rank_move={displacement.mean():.2f}, max_rank_move={displacement.max():.0f}"
    )
