import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score


TARGET = "Will_Buy_EV"
PUBLIC_FRACTION = (0.20 * 286_571) / 668_665
SIMULATIONS = 50_000
SEED = 42
KEPT_BANDS = [(0.54, 0.04), (0.20, 0.06), (0.58, 0.04), (0.47, 0.06)]
NESTED_BANDS = [
    (0.54, 0.02),
    (0.56, 0.02),
    (0.47, 0.03),
    (0.50, 0.03),
    (0.20, 0.03),
    (0.23, 0.03),
    (0.58, 0.02),
    (0.60, 0.02),
]


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


train = pd.read_csv("train.csv")
y = (train[TARGET] == "Yes").astype(np.uint8).to_numpy()
base = rank01(np.load("oof_v37b_stack.npy"))
calibrated = swap_bands(base, KEPT_BANDS)
calibrated_auc = roc_auc_score(y, calibrated)
rng = np.random.default_rng(SEED)
total_pos = int(y.sum())
total_neg = len(y) - total_pos

rows = []
for lower_bound, width in NESTED_BANDS:
    candidate = swap_bands(calibrated, [(lower_bound, width)])
    delta = (roc_auc_score(y, candidate) - calibrated_auc) * 1e5
    half = width / 2
    lower = (calibrated >= lower_bound) & (calibrated < lower_bound + half)
    upper = (calibrated >= lower_bound + half) & (calibrated < lower_bound + width)
    lower_pos = int(y[lower].sum())
    lower_neg = int(lower.sum() - lower_pos)
    upper_pos = int(y[upper].sum())
    upper_neg = int(upper.sum() - upper_pos)

    lp = rng.binomial(lower_pos, PUBLIC_FRACTION, SIMULATIONS)
    ln = rng.binomial(lower_neg, PUBLIC_FRACTION, SIMULATIONS)
    up = rng.binomial(upper_pos, PUBLIC_FRACTION, SIMULATIONS)
    un = rng.binomial(upper_neg, PUBLIC_FRACTION, SIMULATIONS)
    sampled_pos = rng.binomial(total_pos, PUBLIC_FRACTION, SIMULATIONS)
    sampled_neg = rng.binomial(total_neg, PUBLIC_FRACTION, SIMULATIONS)
    simulated_delta = (lp * un - up * ln) / (sampled_pos * sampled_neg) * 1e5
    rows.append(
        {
            "lo": lower_bound,
            "width": width,
            "oof_delta_units": delta,
            "sim_mean": simulated_delta.mean(),
            "sim_sd": simulated_delta.std(),
            "p_positive": (simulated_delta >= 0.5).mean(),
            "p_plus2": (simulated_delta >= 1.5).mean(),
            "lower_rate": lower_pos / lower.sum(),
            "upper_rate": upper_pos / upper.sum(),
        }
    )

result = pd.DataFrame(rows).sort_values(
    ["p_plus2", "oof_delta_units"], ascending=False
)
print(result.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
