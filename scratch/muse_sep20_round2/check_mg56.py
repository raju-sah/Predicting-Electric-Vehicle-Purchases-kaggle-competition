"""Check megayak56 submission.csv vs ds_megayak6 ensemble vs v57."""
import pandas as pd
from scipy.stats import spearmanr

sample = pd.read_csv("sample_submission.csv")
mg = pd.read_csv("scratch/pull_sep19/megayak56/out/submission.csv")
sv = pd.read_csv("scratch/pull_sep16/ds_megayak6/test_six_views.csv")
v57 = pd.read_csv("subs/v57_public_band_swap_exact.csv")["Will_Buy_EV"].to_numpy()
print(mg.shape, list(mg.columns))
print("ids ok:", mg["id"].equals(sample["id"]))
m = mg["Will_Buy_EV"].to_numpy()
e = sv["ensemble"].to_numpy()
print(f"rho(mg56, sixview_ensemble)={spearmanr(m, e)[0]:.7f}")
print(f"rho(mg56, v57)={spearmanr(m, v57)[0]:.7f}")
print(f"rho(sixview_ensemble, v57)={spearmanr(e, v57)[0]:.7f}")
