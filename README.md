# Predicting-Electric-Vehicle-Purchases-kaggle-competition

Kaggle Playground Series **S6E9 — Predicting Electric Vehicle Purchases**.

Binary classification of `Will_Buy_EV` (Yes/No, 17.5% positive rate), scored by **ROC-AUC**.

| | |
|---|---|
| Train rows | 668,000 |
| Test rows | 287,000 |
| Features | 7 numeric + 6 categorical |
| Metric | ROC-AUC |
| Best public LB | **0.94656** (live rank 17) |

## Approach

Every OOF in this repo is produced by the **same fixed 5-fold `StratifiedKFold(shuffle=True, random_state=42)`**.
A score produced under any other fold definition is not comparable and is not treated as valid evidence.

Target encoding is `(df == "Yes").astype(int)`.

Two ideas carry almost all of the signal:

1. **Source-derived features.** The dataset is synthetic and generated from a purchase-probability formula.
   Rebuilding that formula (plus ordinal-worry, total-charger, income×subsidy and concern×subsidy helpers)
   as continuous features is worth more than any generic GBDT tuning.
2. **Rank-space ensembling.** Model predictions are converted to percentile ranks before blending so that
   mismatched calibrations cancel instead of compounding. Prob-space blending was measured and is worse.

Rank correlation between members is what matters, not diversity in the abstract: past
`rho_pred ~= 0.995`, micro-blends are inert. Score differences at the top of the board are real
(`rho ~= 0.99` implies `SD_diff ~= 0.14 sigma`), so only close-parity new members move the needle.

## Layout

| Path | Contents |
|---|---|
| `train_*.py` | One script per experiment, chronological. Baselines, encoding probes, stacking, regime experts, the v35 merged-feature stack. |
| `scratch/` | Research workspace: competitor/kernel pulls, recipe decoders, submission forensics, band-swap builders, muse tooling. |
| `exp_log.csv` | Every attempt ever run, including killed ones, with its OOF and public-LB numbers. |
| `AGENTS.md` | Operating rules: fold gate, submission format, known-dead ends. Read this first. |

Large binaries are **not** committed — `train.csv`, `test.csv`, `*.npy` OOF vectors, `subs/*.csv`
submission files and `external/` are all gitignored. Regenerate or re-download from Kaggle.

## What did not work

Kept here because the negative results are the expensive part of this competition and are
recorded in `AGENTS.md` so they are not retried:

- All-GBM blends and stacks (pairwise correlation > 0.996)
- GMM mode categoricals (-0.0032)
- RandomForest / ExtraTrees (same bias, worse)
- Pseudo-labelling 159k confident test rows
- External `ev-adoption-behavior-and-range-anxiety` rows (rate-matched, no gain)
- Feature-only adversarial validation (0.4989 — confirms no train/test shift)
- Merged-feature solo model: 0.945912 OOF, below the frontier A-view solo

## Submission format

Header exactly `id,Will_Buy_EV`, probabilities, test-id order preserved. Validate with:

```python
assert (pd.read_csv("sample_submission.csv")["id"]
        == pd.read_csv("subs/<file>.csv")["id"]).all()
```

```bash
kaggle competitions submit -c playground-series-s6e9 -f subs/<file>.csv -m "<ver> <model> OOF <score>"
kaggle competitions submissions -c playground-series-s6e9   # publicScore lands after ~60s
```