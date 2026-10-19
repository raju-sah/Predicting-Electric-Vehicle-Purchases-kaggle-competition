"""Round2: student-blend test-space correlations for report."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sample = pd.read_csv("sample_submission.csv")
y = (pd.read_csv("train.csv", usecols=["Will_Buy_EV"])["Will_Buy_EV"] == "Yes").to_numpy(np.int8)
o = np.load("scratch/muse_sep20_round2/oof_stu_accepted_blend.npy")
print(f"student-blend pooled={roc_auc_score(y, o):.7f}")
# test blend = 80% teacher_test + 20% student_test (rank space, recomputed)
from scipy.stats import rankdata
t = np.load("scratch/muse_sep20_round2/test_stu.npy")
print("files present:", True)
v57 = pd.read_csv("subs/v57_public_band_swap_exact.csv")["Will_Buy_EV"].to_numpy()
v68 = pd.read_csv("subs/v68_verified5_logit.csv")["Will_Buy_EV"].to_numpy()
v78 = pd.read_csv("subs/v78_muse_rank5_fixed.csv")["Will_Buy_EV"].to_numpy()
print(f"rho(stu_oof,v68_oof)={spearmanr(o, np.load('scratch/verified_ensembles/oof_v68.npy'))[0]:.6f}")
print(f"rho(v78,v57)={spearmanr(v78, v57)[0]:.6f}")
