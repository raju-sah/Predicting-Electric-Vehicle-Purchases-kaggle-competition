"""Test row-index periodicity as a non-feature source of signal.

The competition IDs are consecutive and train/test are adjacent ranges. This
uses only rates learned from the training portion of each prescribed fold for
id % modulus, so any score above chance must repeat across independent rows
with the same residue. It does not retrain or blend additional GBMs.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

TARGET = "Will_Buy_EV"
MODULI = (64, 128, 256, 512, 1000, 1024, 2048, 4096, 8192, 10_000)
SMOOTHING = 100.0

train = pd.read_csv("train.csv", usecols=["id", TARGET])
test = pd.read_csv("test.csv", usecols=["id"])
y = (train[TARGET] == "Yes").astype(np.int8).to_numpy()
ids = train["id"].to_numpy()
test_ids = test["id"].to_numpy()
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
splits = list(cv.split(ids, y))
prior = y.mean()


def residue_oof(modulus: int) -> tuple[np.ndarray, np.ndarray]:
    """Return fold-safe target rates and full-train test rates for one modulus."""
    oof = np.empty(len(y), dtype=np.float64)
    for tr_idx, va_idx in splits:
        residue = ids[tr_idx] % modulus
        counts = np.bincount(residue, minlength=modulus)
        positives = np.bincount(residue, weights=y[tr_idx], minlength=modulus)
        rates = (positives + SMOOTHING * prior) / (counts + SMOOTHING)
        oof[va_idx] = rates[ids[va_idx] % modulus]
    counts = np.bincount(ids % modulus, minlength=modulus)
    positives = np.bincount(ids % modulus, weights=y, minlength=modulus)
    rates = (positives + SMOOTHING * prior) / (counts + SMOOTHING)
    return oof, rates[test_ids % modulus]


results = []
for modulus in MODULI:
    oof, pred = residue_oof(modulus)
    auc = roc_auc_score(y, oof)
    print(f"id % {modulus:>5}: OOF AUC {auc:.7f}", flush=True)
    results.append((auc, modulus, oof, pred))

best_auc, best_modulus, best_oof, best_pred = max(results, key=lambda x: x[0])
np.save("oof_id_structure.npy", best_oof)
pd.DataFrame({"id": test_ids, TARGET: best_pred}).to_csv(
    "subs/id_structure_probe.csv", index=False
)
print(f"BEST id % {best_modulus}: OOF AUC {best_auc:.7f}")

# This diagnostic only assesses complementarity. It is not a submission
# candidate unless the resulting OOF clears AGENTS.md's hard gate.
lgb_oof = np.load("oof_lgb.npy")
for weight in (0.005, 0.01, 0.02, 0.05, 0.10):
    blend = (1.0 - weight) * lgb_oof + weight * best_oof
    print(
        f"LGB + {weight:.3f} id structure: OOF AUC {roc_auc_score(y, blend):.7f}",
        flush=True,
    )
