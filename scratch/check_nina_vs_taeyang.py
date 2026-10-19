import pandas as pd
import numpy as np

p_nina = pd.read_csv("scratch/kernels_pulled/nina_h4_out/submission.csv")["Will_Buy_EV"].values
p_taeyang = pd.read_csv("scratch/kernels_pulled/taeyangg4_094650_out/submission.csv")["Will_Buy_EV"].values

diff = np.abs(p_nina - p_taeyang)
print(f"Max difference between nina_h4 and taeyang: {diff.max():.8f}")
print(f"Mean difference: {diff.mean():.8f}")
print(f"Are they identical? {np.array_equal(p_nina, p_taeyang)}")
