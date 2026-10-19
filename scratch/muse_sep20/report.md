# S6E9 muse_sep20 report — 2026-09-20

Worker: muse (dispatched task_20f312caad26). Nothing was submitted to Kaggle by
this worker. All four candidates are new files (v75–v78); no existing
submission, OOF array, or audit file was overwritten. exp_log.csv was appended
only (5 rows); prior rows untouched.

## 1. Ranked candidates (all complete fixed-five-fold OOF, gate 0.94230)

| Rank | File | Method | Pooled OOF / mean fold | Expected public LB | Go / No-go |
|---|---|---|---|---|---|
| 1 | subs/v78_muse_rank5_fixed.csv | rank blend mem40/lgb6-15/xgb5-15/catA15/hgbB15 | 0.9460654 / 0.9460703 | 0.94630–0.94645 | CONDITIONAL-GO (backup frontier probe) |
| 2 | subs/v77_muse_prob5_fixed.csv | probability blend, same weights | 0.9460407 / 0.9460456 | 0.94630–0.94645 | CONDITIONAL-GO (only if v78 used first) |
| 3 | subs/v75_muse_catA_solo.csv | CatBoost ordered boosting solo | 0.9433034 / 0.9433106 | 0.94300–0.94350 | NO-GO standalone (evidence/diversity member) |
| 4 | subs/v76_muse_hgbB_solo.csv | HGB native-cats solo | 0.9432323 / 0.9432399 | 0.94300–0.94350 | NO-GO standalone (evidence/diversity member) |

No candidate has a demonstrated path to beat 0.94656, let alone 0.94675.
First place is not guaranteed by any of these. The honest expectation, anchored
on v68 (complete OOF 0.94627 → public 0.94645), is that v78/v77 land around
0.9463–0.94645: frontier-adjacent probes, not breakthroughs. Submit v78 only if
a slot can be spent on a principled diversity probe; never more than one of
this family per day.

## 2. Novelty (specific new arguments, not killed-idea retries)

- CatBoost-A: early `cbm_native` (0.9414) used raw features only. This is the
  first CatBoost trained on the proven digit-frame representation plus 8 novel
  fit-free ratios/flags, with native categorical handling and ordered boosting —
  a learner family absent from the entire verified pool (LGBM/XGB only).
- HGB-B: only a raw HGB-default baseline (v01, 0.9409) ever existed. First HGB
  on the digit frame with native categorical support (constructor
  `categorical_features`, sklearn 1.9).
- The 8 added features (income/age, total chargers, chargers/car, commute×cars,
  income×env, env×subsidy, work-access, age×cars, plus 30k/millionaire/dead-zone/
  env-hater/home flags) are fit-free and disjoint from the 4 killed interactions
  (charging_access, anx_commute, income_per_car, home_sub — all excluded).
- Ensemble weights [0.40, 0.15, 0.15, 0.15, 0.15] were fixed in
  `build_candidates.py` (written before either new model finished) — the
  reported OOF is confirmatory, not hill-climbed (contrast v56/v37b, whose
  weights were selected on the OOF labels).

## 3. Leak risks

- Both new models use only fit-free features plus in-fold native categorical
  handling. No target encodings, no external rows, no test-label use, no
  band-swap, no JZ-anchor lineage (JZ has no train-fold predictions, so no JZ
  blend can carry complete comparable OOF — that lineage was excluded by design).
- lgb6/xgb5/memory components carry published-source provenance (see
  scratch/research_next/najiama_provenance.md; memory fold/labels re-verified
  in this task). Component choice reuses public knowledge; OOF intervals do not
  adjust for selection. Stated plainly: these OOFs are descriptive of the fixed
  folds, not an independent generalization guarantee.

## 4. Full validation (re-verified from disk, script validate_final.py)

- OOF evidence: scratch/muse_sep20/oof_v75.npy … oof_v78.npy; pooled AUCs
  recomputed from disk match scratch/muse_sep20/manifest_muse.json to 1e-12.
- Per-fold tables:
  - v75: 0.94222 / 0.94307 / 0.94422 / 0.94372 / 0.94334
  - v76: 0.94211 / 0.94298 / 0.94417 / 0.94365 / 0.94329
  - v77: 0.94504 / 0.94573 / 0.94696 / 0.94636 / 0.94613
  - v78: 0.94506 / 0.94576 / 0.94697 / 0.94639 / 0.94616
- Submission files: header exactly `id,Will_Buy_EV`; 286,571 rows; ids unique,
  equal and order-equal to sample_submission.csv; all finite; all in [0, 1];
  sha256 matches manifest (see §6). All four exceed gate 0.94230.

## 5. Correlations / diversity (test space, Spearman)

- New learners vs frontier: ρ(catA, memory) = 0.9890, ρ(hgbB, memory) = 0.9883 —
  comparable to digit-frame's 0.9885 decorrelation, i.e. genuinely additive
  diversity, not a ρ≈0.999 clone (v66–v68 are mutually ρ≈1.0).
- catA vs hgbB: ρ = 0.9931 (share features, differ in learner).
- Blends vs anchors: ρ(v78, v68) = 0.99976, ρ(v78, v57) = 0.99798 —
  the blends live in the frontier family; the solos (ρ≈0.988 vs v57) are the
  diversity reservoir for future ensembles, not standalone submissions.

## 6. Exact file hashes (sha256)

- subs/v75_muse_catA_solo.csv:
  `8c445f849c631abfb98ebe7fd61e3319b225ad8da3145f3fbc9158b98e2d5aed`
- subs/v76_muse_hgbB_solo.csv:
  `5966cffec7f87e7f20746744f7d4a5fd464c2c24d9788aba8cc515daf1c1ebef`
- subs/v77_muse_prob5_fixed.csv:
  `efb3f24a4f5bd6c3a3f180542d489c597a2b3c77a4a35276b140016ca74ce61a`
- subs/v78_muse_rank5_fixed.csv:
  `48449671ad874de442fdd465e3ee3c3f1813f256cac04a682d8ac36b1d159c98`

## 7. Documented alternates (NOT in subs/, kept staged, hashes in staging/)

scratch/muse_sep20/staging/ holds four fully validated 4-way blends without
catA (mem45/lgb20/xgb20/hgb15): logit 0.9462450, rank 0.9462240, prob 0.9462151,
plus the hgbB solo copy. Their complete OOFs are marginally higher than v77/v78
(the two weaker new members dilute the 5-way mean). If the coordinator prefers
peak complete-OOF over new-learner representation, promote at most one staged
blend under a fresh vNN name — do not relabel or overwrite v75–v78.

## 8. LB / budget context (read-only, Sep 20)

Board top: Thimsen 0.94675, Deotte 0.94672, yalpha 0.94667, BOB 0.94666. Our best
0.94656 (v57–v59). Today: v71 0.94646, v72 0.94650 (both below best) — 2 of 10
daily submissions spent at check time. This worker submitted nothing.

## 9. Reproducibility

- scratch/muse_sep20/train_hgbB.py → oof_hgbB.npy, test_hgbB.npy, run_hgbB.json
- scratch/muse_sep20/train_catA.py → oof_catA.npy, test_catA.npy, run_catA.json
- scratch/muse_sep20/build_candidates.py → subs/v75–v78, oof_v75–v78.npy,
  manifest_muse.json (weights pre-registered)
- scratch/muse_sep20/validate_final.py, audit_context.py, check_hgb_div.py,
  dryrun_fallback.py, stage_provisionals.py, append_log.py, logs log_hgbB.txt /
  log_catA.txt. Runtimes: HGB ~8 min, CatBoost ~88 min (8 CPU, no GPU).

## 10. Blocker / what is left

Four qualifying candidates are delivered, so there is no gate blocker. The
scientific blocker stands: no trained-from-scratch candidate in this family
beats the public-tuned JZ band-swap lineage (0.94656), because that lineage's
+0.00005 comes from public-split information no fixed-fold OOF can reproduce.
Closing the remaining +0.00019 to #1 needs either a genuinely stronger anchor
(a better base ranking, not a re-blend) or new public-split insight — neither
was found in this run or in the Sep-19 public refresh.
