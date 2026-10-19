import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
import pandas as pd

a = np.load("scratch/oof_digit_fix.npy")
b = np.load("oof_digit_features.npy")
y = (pd.read_csv("train.csv", usecols=["Will_Buy_EV"])["Will_Buy_EV"] == "Yes").astype(int).to_numpy()
print(f"fix OOF={roc_auc_score(y, a):.7f} base OOF={roc_auc_score(y, b):.7f}")
print(f"spearman fix-vs-base: {spearmanr(a, b).statistic:.6f}")
print(f"mean|diff|: {np.mean(np.abs(a - b)):.6f}")
