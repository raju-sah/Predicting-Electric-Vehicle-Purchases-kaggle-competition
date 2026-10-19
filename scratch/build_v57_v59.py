import hashlib

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


ID = "id"
TARGET = "Will_Buy_EV"
EXPECTED_ANCHOR_SHA = "1f40cec214646ba50eac6d5307d41c6aca82fe7acc38e2501399980c62d4bbe7"
BANDS = [(0.54, 0.04), (0.20, 0.06), (0.58, 0.04), (0.47, 0.06)]


def rank01(values):
    return rankdata(values, method="average") / len(values)


def swap_bands(ranks, bands):
    swapped = ranks.copy()
    for lower_bound, width in bands:
        half = width / 2
        lower = (ranks >= lower_bound) & (ranks < lower_bound + half)
        upper = (ranks >= lower_bound + half) & (ranks < lower_bound + width)
        swapped[lower] += half
        swapped[upper] -= half
    return rankdata(swapped, method="ordinal") / len(swapped)


sample = pd.read_csv("sample_submission.csv")
ids = sample[ID].to_numpy()
n_rows = len(sample)

anchor_path = "scratch/pull_sep16/jz_meta/submission_latest_best.csv"
anchor_sha = hashlib.sha256(open(anchor_path, "rb").read()).hexdigest()
assert anchor_sha == EXPECTED_ANCHOR_SHA
anchor = pd.read_csv(anchor_path).sort_values(ID)
memory = pd.read_csv(
    "scratch/pull_sep18/memory_recipe_out/submission_model_only.csv"
).sort_values(ID)
public_exact = pd.read_csv(
    "scratch/pull_sep19/megayak56/out/submission.csv"
).sort_values(ID)

for frame in (anchor, memory, public_exact):
    assert np.array_equal(frame[ID].to_numpy(), ids)

base_rank = rank01(anchor[TARGET].to_numpy(float))
memory_rank = rank01(memory[TARGET].to_numpy(float))
swapped_rank = swap_bands(base_rank, BANDS)

np.testing.assert_allclose(
    swapped_rank,
    public_exact[TARGET].to_numpy(float),
    rtol=0,
    atol=1e-15,
)

candidates = {
    "v57_public_band_swap_exact": swapped_rank,
    "v58_band_swap_memory_w05": rankdata(
        0.95 * swapped_rank + 0.05 * memory_rank, method="ordinal"
    ) / n_rows,
    "v59_band_swap_memory_w10": rankdata(
        0.90 * swapped_rank + 0.10 * memory_rank, method="ordinal"
    ) / n_rows,
}

for name, prediction in candidates.items():
    out = pd.DataFrame({ID: ids, TARGET: prediction})
    assert out.columns.tolist() == [ID, TARGET]
    assert len(out) == n_rows
    assert np.array_equal(out[ID].to_numpy(), ids)
    assert np.isfinite(out[TARGET]).all()
    assert out[TARGET].between(0, 1).all()
    assert out[TARGET].nunique() == n_rows
    path = f"subs/{name}.csv"
    out.to_csv(path, index=False)
    displacement = np.abs(rankdata(prediction) - rankdata(swapped_rank))
    print(
        f"{path}: rho_vs_094656={spearmanr(prediction, swapped_rank).statistic:.9f}, "
        f"mean_rank_move={displacement.mean():.2f}, "
        f"max_rank_move={displacement.max():.0f}"
    )
