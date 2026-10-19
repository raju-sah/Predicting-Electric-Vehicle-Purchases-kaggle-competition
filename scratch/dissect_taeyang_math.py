import pandas as pd
import numpy as np
from scipy.stats import rankdata

test = pd.read_csv("test.csv")
TARGET = "Will_Buy_EV"

# Let's inspect the exact components Taeyang used
p49 = pd.read_csv("scratch/kernels_pulled/jazivxt_zoom_out/submission.csv")[TARGET].to_numpy(dtype=float)
p_rmlp = pd.read_csv("scratch/realmlp_out/submission.csv")[TARGET].to_numpy(dtype=float)

# Wait, what was p_ctb in Taeyang's run? Let's check where ctb was found in Taeyang's notebook!
with open("scratch/kernels_pulled/taeyangg4_094650_out/s6e9-094650-multi-paradigm-lexsort-master.log") as f:
    log_content = f.read()
print("Log output from Taeyang:")
print(log_content)
