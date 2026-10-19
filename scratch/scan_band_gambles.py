import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata


TARGET = "Will_Buy_EV"
PUBLIC_ROWS = 0.20 * 286_571
KEPT_BANDS = [(0.54, 0.04), (0.20, 0.06), (0.58, 0.04), (0.47, 0.06)]


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
ranks = swap_bands(rank01(np.load("oof_v37b_stack.npy")), KEPT_BANDS)
sample_fraction = PUBLIC_ROWS / len(y)
total_pos = int(y.sum())
total_neg = len(y) - total_pos
denominator = (sample_fraction * total_pos) * (sample_fraction * total_neg)

rows = []
for width in np.arange(0.02, 0.201, 0.01):
    for lower_bound in np.arange(0.10, 0.901, 0.01):
        if lower_bound + width > 0.99:
            continue
        half = width / 2
        lower = (ranks >= lower_bound) & (ranks < lower_bound + half)
        upper = (ranks >= lower_bound + half) & (ranks < lower_bound + width)
        lp = int(y[lower].sum())
        ln = int(lower.sum() - lp)
        up = int(y[upper].sum())
        un = int(upper.sum() - up)

        expected_numerator = sample_fraction**2 * (lp * un - up * ln)
        mean_units = expected_numerator / denominator * 1e5

        def product_variance(first_count, second_count):
            first_mean = sample_fraction * first_count
            second_mean = sample_fraction * second_count
            first_var = first_count * sample_fraction * (1 - sample_fraction)
            second_var = second_count * sample_fraction * (1 - sample_fraction)
            return (
                first_var * second_var
                + first_var * second_mean**2
                + second_var * first_mean**2
            )

        numerator_var = product_variance(lp, un) + product_variance(up, ln)
        sd_units = np.sqrt(numerator_var) / denominator * 1e5
        rows.append(
            {
                "lo": lower_bound,
                "width": width,
                "mean": mean_units,
                "sd": sd_units,
                "p_plus1": norm.sf((0.5 - mean_units) / sd_units),
                "p_plus5": norm.sf((4.5 - mean_units) / sd_units),
                "p_plus10": norm.sf((9.5 - mean_units) / sd_units),
                "p_plus19": norm.sf((18.5 - mean_units) / sd_units),
            }
        )

result = pd.DataFrame(rows)
eligible = result[result["mean"] >= -5.0].copy()
for column in ("p_plus19", "p_plus10", "p_plus5", "p_plus1"):
    print(f"\nTop by {column}")
    print(
        eligible.sort_values(column, ascending=False)
        .head(20)
        .to_string(index=False, float_format=lambda value: f"{value:.6f}")
    )
