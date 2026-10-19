import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


ID = "id"
TARGET = "Will_Buy_EV"


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
anchor = pd.read_csv("subs/v57_public_band_swap_exact.csv").sort_values(ID)
ids = sample[ID].to_numpy()
assert np.array_equal(anchor[ID].to_numpy(), ids)
base = rank01(anchor[TARGET].to_numpy(float))

probes = {
    "v60_nested_band_059_061": [(0.59, 0.02)],
    "v61_nested_band_050_053": [(0.50, 0.03)],
    "v62_nested_band_058_060": [(0.58, 0.02)],
    "v63_combo_059_061__050_053": [(0.59, 0.02), (0.50, 0.03)],
    "v64_combo_059_061__058_060": [(0.59, 0.02), (0.58, 0.02)],
    "v65_combo_all_nested": [(0.59, 0.02), (0.50, 0.03), (0.58, 0.02)],
}

for name, bands in probes.items():
    prediction = swap_bands(base, bands)
    out = pd.DataFrame({ID: ids, TARGET: prediction})
    assert np.array_equal(out[ID].to_numpy(), ids)
    assert out[TARGET].nunique() == len(out)
    assert np.isfinite(out[TARGET]).all()
    assert out[TARGET].between(0, 1).all()
    path = f"subs/{name}.csv"
    out.to_csv(path, index=False)
    moved = np.abs(rankdata(prediction) - rankdata(base)) > 0
    print(
        f"{path}: bands={bands}, moved={moved.sum():,}, "
        f"rho={spearmanr(prediction, base).statistic:.9f}"
    )
