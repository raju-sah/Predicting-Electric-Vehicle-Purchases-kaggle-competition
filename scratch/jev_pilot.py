"""Jev pilot: 2000 train + 1000 test rows, one request/row, five nouls.
Gate: direct AUC >= 0.82 and novelty (rho < 0.995 vs anchors) or KILL.
Cache: scratch/jev_pilot_cache.json — reruns free.
"""
import hashlib
import json
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

GW_URL = "https://ai-gateway.vercel.sh/typesafe/v1/systemone"
MODEL = "typesafe-ai/jev"
CACHE = "scratch/jev_pilot_cache.json"
N_TRAIN, N_TEST, WORKERS = 300, 200, 4

QUESTIONS = {
    "will_buy": {"type": "noul", "instructions": "Will this person buy an EV?",
                 "criteria": {"true": "Profile indicates likely EV purchase",
                              "false": "Profile indicates unlikely EV purchase"}},
    "afford": {"type": "noul", "instructions": "Can this person afford an EV given income, cars owned and subsidy?",
               "criteria": {"true": "Income/subsidy clearly supports EV purchase",
                            "false": "Budget clearly blocks EV purchase"}},
    "commute_fit": {"type": "noul", "instructions": "Does daily commute plus nearby charging make EV practical?",
                    "criteria": {"true": "Commute short or charging abundant",
                                 "false": "Long commute with poor charging access"}},
    "env_intent": {"type": "noul", "instructions": "Does environmental concern signal intent to buy EV?",
                   "criteria": {"true": "High concern pushing toward EV",
                                "false": "Low concern, no green motive"}},
    "home_block": {"type": "noul", "instructions": "Does lack of home charging block EV purchase?",
                   "criteria": {"true": "No home charging and no workable alternative",
                                "false": "Home charging possible or alternatives cover it"}},
}

COLS = ["Age", "Annual_Income_USD", "Daily_Commute_km", "Number_of_Cars_Owned",
        "Charging_Stations_Near_Home", "Charging_Stations_Near_Work",
        "Environmental_Concern_Level", "Gender", "City_Type", "Current_Car_Type",
        "Home_Charging_Possible", "Subsidy_Available", "Range_Anxiety_Level"]


def row_state(r):
    out = {}
    for c in COLS:
        v = r[c]
        out[c] = float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v
    return out


def call_jev(state, key, tries=4):
    body = json.dumps({"model": MODEL, "state": state, "questions": QUESTIONS}).encode()
    for a in range(tries):
        req = urllib.request.Request(GW_URL, data=body,
                                     headers={"Authorization": f"Bearer {key}",
                                              "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                ans = json.load(resp)["answers"]
            return {k: float(v["noul"]) for k, v in ans.items()}, None
        except Exception as e:  # noqa: BLE001 — 429/529 retry with backoff
            err = str(e)
            if ("429" in err or "529" in err) and a < tries - 1:
                time.sleep(2 ** a)
                continue
            return None, err
    return None, "retries exhausted"


def main():
    key = os.environ["AI_GATEWAY_API_KEY"]
    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    y_all = (train["Will_Buy_EV"] == "Yes").astype(int).to_numpy()
    rng = np.random.default_rng(42)
    pos = np.where(y_all == 1)[0]
    neg = np.where(y_all == 0)[0]
    n_pos = int(N_TRAIN * 0.175)
    tr_idx = np.sort(np.concatenate([rng.choice(pos, n_pos, replace=False),
                                     rng.choice(neg, N_TRAIN - n_pos, replace=False)]))
    te_idx = np.sort(rng.choice(len(test), N_TEST, replace=False))
    try:
        cache = json.load(open(CACHE))
    except (FileNotFoundError, json.JSONDecodeError):
        cache = {}
    jobs = []
    for i in tr_idx:
        r = train.iloc[i]
        h = hashlib.md5(json.dumps(row_state(r), sort_keys=True).encode()).hexdigest()
        jobs.append((f"tr:{i}", h, row_state(r), int(y_all[i])))
    for j in te_idx:
        r = test.iloc[j]
        h = hashlib.md5(json.dumps(row_state(r), sort_keys=True).encode()).hexdigest()
        jobs.append((f"te:{j}", h, row_state(r), -1))
    todo = [jb for jb in jobs if jb[1] not in cache]
    print(f"rows={len(jobs)} cached={len(jobs) - len(todo)} todo={len(todo)}", flush=True)

    def fetch(jb):
        tag, h, state, _ = jb
        out, err = call_jev(state, key)
        return tag, h, out, err

    errs, t0, in_tok, done = 0, time.time(), 0, 0
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for tag, h, out, err in pool.map(fetch, todo):
            if out is None:
                errs += 1
            else:
                cache[h] = out
                in_tok += 350
            done += 1
            if done % 20 == 0:  # ponytail: flush incrementally so timeouts keep progress
                json.dump(cache, open(CACHE, "w"))
    json.dump(cache, open(CACHE, "w"))
    dt = time.time() - t0
    print(f"done in {dt:.0f}s errors={errs} est_in_tokens~{in_tok}", flush=True)

    labels, direct = [], []
    for tag, h, _, y in jobs:
        if tag.startswith("tr:") and h in cache:
            labels.append(y)
            direct.append(cache[h]["will_buy"])
    labels, direct = np.array(labels), np.array(direct)
    auc = roc_auc_score(labels, direct)
    print(f"Jev direct AUC on {len(labels)} train rows: {auc:.4f}", flush=True)
    for q in QUESTIONS:
        col = np.array([cache[jb[1]][q] for jb in jobs if jb[0].startswith("tr:") and jb[1] in cache])
        print(f"  {q}: AUC={roc_auc_score(labels, col):.4f}", flush=True)
    # novelty vs anchors on test sample
    v81 = pd.read_csv("subs/v81_mega56_champ_50_50.csv").set_index("id")["Will_Buy_EV"]
    v68 = pd.read_csv("subs/v68_verified5_logit.csv").set_index("id")["Will_Buy_EV"]
    tids = test.iloc[te_idx]["id"].to_numpy()
    jd = np.array([cache[jb[1]]["will_buy"] for jb in jobs if jb[0].startswith("te:") and jb[1] in cache])
    a81 = v81.reindex(tids).to_numpy(dtype=float)
    a68 = np.clip(v68.reindex(tids).to_numpy(dtype=float), 1e-9, 1 - 1e-9)
    print(f"rho vs v81: {spearmanr(jd, a81).statistic:.5f}  rho vs v68: {spearmanr(jd, a68).statistic:.5f}", flush=True)
    print("GATE:", "PASS pilot → disagreement-slice scale" if auc >= 0.82 else "KILL — below 0.82", flush=True)


if __name__ == "__main__":
    main()
