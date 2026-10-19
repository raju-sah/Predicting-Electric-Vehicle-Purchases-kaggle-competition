"""Score Jev pilot from cache only — no API calls. Aligns strictly on cached rows."""
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

COLS = ["Age", "Annual_Income_USD", "Daily_Commute_km", "Number_of_Cars_Owned",
        "Charging_Stations_Near_Home", "Charging_Stations_Near_Work",
        "Environmental_Concern_Level", "Gender", "City_Type", "Current_Car_Type",
        "Home_Charging_Possible", "Subsidy_Available", "Range_Anxiety_Level"]
QS = ["will_buy", "afford", "commute_fit", "env_intent", "home_block"]


def h(r):
    d = {}
    for c in COLS:
        v = r[c]
        d[c] = float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v
    return hashlib.md5(json.dumps(d, sort_keys=True).encode()).hexdigest()


cache = json.load(open("scratch/jev_pilot_cache.json"))
print(f"cache entries: {len(cache)}")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
y_all = (train["Will_Buy_EV"] == "Yes").astype(int).to_numpy()
rng = np.random.default_rng(42)
pos = np.where(y_all == 1)[0]
neg = np.where(y_all == 0)[0]
n_pos = int(300 * 0.175)
tr_idx = np.sort(np.concatenate([rng.choice(pos, n_pos, replace=False),
                                 rng.choice(neg, 300 - n_pos, replace=False)]))
te_idx = np.sort(rng.choice(len(test), 200, replace=False))

labels, cols = [], {q: [] for q in QS}
for i in tr_idx:
    key = h(train.iloc[i])
    if key in cache:
        labels.append(int(y_all[i]))
        for q in QS:
            cols[q].append(cache[key][q])
labels = np.array(labels)
print(f"train rows scored: {len(labels)}/300")
for q in QS:
    print(f"  {q}: AUC={roc_auc_score(labels, np.array(cols[q])):.4f}")

v81 = pd.read_csv("subs/v81_mega56_champ_50_50.csv").set_index("id")["Will_Buy_EV"]
v68 = pd.read_csv("subs/v68_verified5_logit.csv").set_index("id")["Will_Buy_EV"]
jd, a81, a68 = [], [], []
for j in te_idx:
    key = h(test.iloc[j])
    tid = test.iloc[j]["id"]
    if key in cache and tid in v81.index and tid in v68.index:
        jd.append(cache[key]["will_buy"])
        a81.append(float(v81.loc[tid]))
        a68.append(float(np.clip(v68.loc[tid], 1e-9, 1 - 1e-9)))
jd, a81, a68 = map(np.array, (jd, a81, a68))
print(f"test rows scored: {len(jd)}/200")
print(f"rho vs v81: {spearmanr(jd, a81).statistic:.5f}  rho vs v68: {spearmanr(jd, a68).statistic:.5f}")
aucs = {q: roc_auc_score(labels, np.array(cols[q])) for q in QS}
gate = "PASS" if aucs["will_buy"] >= 0.82 else "KILL"
print(f"GATE direct>=0.82: {gate} ({aucs['will_buy']:.4f})")
