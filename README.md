# Predicting-Electric-Vehicle-Purchases-kaggle-competition

Kaggle Playground Series **S6E9 — Predicting Electric Vehicle Purchases**.

Binary classification of `Will_Buy_EV` (Yes/No, 17.5% positive rate), scored by **ROC-AUC**.

| | |
|---|---|
| Train rows | 668,000 |
| Test rows | 287,000 |
| Features | 7 numeric + 6 categorical |
| Metric | ROC-AUC |

## Leaderboard

Final standings, competition completed. Both boards have the same 62 entries.

| Board | Rank | Score | Entries |
|---|---|---|---|
| **Public** | **353** / 3,575 | **0.94657** | 62 |
| **Private** | **506** | **0.94542** | 62 |

→ https://www.kaggle.com/competitions/playground-series-s6e9/leaderboard?search=raju+sah

Private is computed on ~80% of the test set. Both rows were read back from the Kaggle CLI
leaderboard download and cross-checked against the competition UI.

Best submission: `v81_mega56_champ_50_50.csv` — 0.94657 public / 0.94542 private. It is a 50/50
rank blend of the megayak six-view pool against the champion anchor, carried over the proven
four-band half-swap. Top of the final board is Team Alicia at 0.94945 public.

Two things are worth reading off that table rather than hiding:

- **Public rank 353 from a score many teams matched.** A tight cluster sits at 0.94657, so the
  public number was never a clean separation — dozens of teams converged on the same band-swap
  neighbourhood and tie-break by submission order alone.
- **Private landed 153 places below public.** The band swap was tuned on the public split, and
  that is exactly the overfit signature: it bought public score and gave it back on private.
  The fold-42 OOF gate in `AGENTS.md` exists precisely because of this, and it should have been
  weighted harder than it was.

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