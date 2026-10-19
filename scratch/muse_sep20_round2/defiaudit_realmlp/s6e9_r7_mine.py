#!/usr/bin/env python3
"""S6E9 r7 mining v3 — vectorized band mining (target is Yes/No string)."""
import numpy as np
import pandas as pd

BASE = "/home/z/my-project/s6e9"
TARGET = "Will_Buy_EV"

tr = pd.read_csv(f"{BASE}/data/train.csv")
y_raw = tr[TARGET].astype(str).str.strip().str.lower()
y = (y_raw == "yes").astype(float).to_numpy()
g = y.mean()
print(f"train={len(tr)} global rate={g:.5f}", flush=True)

inc = tr["Annual_Income_USD"].to_numpy(float)
km = tr["Daily_Commute_km"].to_numpy(float)
age = tr["Age"].to_numpy(float)
sub = tr["Subsidy_Available"].astype(str).to_numpy()
anx = tr["Range_Anxiety_Level"].astype(str).to_numpy()
hc = tr["Home_Charging_Possible"].astype(str).to_numpy()
ct = tr["City_Type"].astype(str).to_numpy()


def band_table(x, step, name, min_n=400, min_dev=0.02):
    lo_idx = np.floor(x / step).astype(int)
    df = pd.DataFrame({"bin": lo_idx, "y": y})
    agg = df.groupby("bin")["y"].agg(["mean", "count"])
    agg["dev"] = agg["mean"] - g
    agg["lo"] = (agg.index * step).astype(float)
    agg["hi"] = ((agg.index + 1) * step).astype(float)
    sel = agg[(agg["count"] >= min_n) & (agg["dev"].abs() >= min_dev)]
    print(f"\n--- {name} (step={step}, n>={min_n}, |dev|>={min_dev}) ---", flush=True)
    print(sel[["lo", "hi", "count", "dev"]].to_string(index=False))
    return sel


band_table(inc, 2000, "income $2k", min_dev=0.025)
band_table(km, 5, "commute 5km", min_dev=0.02)
band_table(age, 5, "age 5y", min_dev=0.02)

print("\n--- income $10k x Subsidy ---", flush=True)
df = pd.DataFrame({"bin": np.floor(inc / 10000).astype(int), "sub": sub, "y": y})
agg = df.groupby(["bin", "sub"])["y"].agg(["mean", "count"]).reset_index()
agg["dev"] = agg["mean"] - g
print(agg[(agg["count"] >= 300) & (agg["dev"].abs() >= 0.03)].to_string(index=False))

print("\n--- Range_Anxiety x Subsidy ---", flush=True)
df = pd.DataFrame({"anx": anx, "sub": sub, "y": y})
agg = df.groupby(["anx", "sub"])["y"].agg(["mean", "count"]).reset_index()
agg["dev"] = agg["mean"] - g
print(agg.to_string(index=False))

print("\n--- Home_Charging x City_Type ---", flush=True)
df = pd.DataFrame({"hc": hc, "ct": ct, "y": y})
agg = df.groupby(["hc", "ct"])["y"].agg(["mean", "count"]).reset_index()
agg["dev"] = agg["mean"] - g
print(agg.to_string(index=False))
