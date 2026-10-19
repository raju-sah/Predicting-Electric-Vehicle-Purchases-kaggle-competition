# S6E9 muse_sep20 candidate report — 2026-09-20 (DRAFT skeleton)

## Staged provisionals (scratch/muse_sep20/staging/, audited 2026-09-20)
All four pass header/row/id/finite/bounds checks with complete fixed-fold OOF.
Full fold tables + sha256 in staging/manifest_staging.json.

| Stage | Content | Pooled OOF | Mean fold | Test rho vs v57 / v68 |
|---|---|---|---|---|
| p75 | hgbB solo | 0.9432323 | 0.9432399 | 0.98754 / 0.98859 |
| p76 | prob blend mem45/lgb20/xgb20/hgb15 | 0.9462151 | 0.9462198 | 0.99763 / 0.99927 |
| p77 | rank blend same weights | 0.9462240 | 0.9462287 | 0.99797 / 0.99973 |
| p78 | logit blend same weights | 0.9462450 | 0.9462497 | 0.99798 / 0.99976 |

Weights fixed a priori; OOF confirmatory. CatBoost-A verdict pending: if its
pooled OOF clears ~0.943 with useful diversity, final v75-v78 become 5-comp
blends + hgbB solo; else these four promote as-is.
| Rank | File | Complete OOF (pooled / mean) | Novelty | Expected LB | Go/No-go |
|---|---|---|---|---|---|
| 1 | subs/v77_muse_prob5_fixed.csv | TBD | 5-way fixed-weight prob ensemble incl. 2 new learners | TBD | TBD |
| 2 | subs/v78_muse_rank5_fixed.csv | TBD | rank-space twin | TBD | TBD |
| 3 | subs/v75_muse_catA_solo.csv | TBD | first CatBoost-ordered-boosting on digit frame | TBD | TBD |
| 4 | subs/v76_muse_hgbB_solo.csv | TBD | first HGB-native-cats on digit frame | TBD | TBD |

Gate: complete fixed-five-fold OOF > 0.94230. No submission recommended below gate.
No file submitted to Kaggle by this worker. DO NOT submit without coordinator review.

## Novelty (vs killed list)
- CatBoost ordered boosting + digit frame: early cbm_native lacked digit/memory
  features; new specific argument, not a retry.
- HGB: only raw HGB-default baseline existed; first HGB on digit frame with
  native categoricals.
- 8 ratio/flag features are fit-free and disjoint from the 4 killed interactions
  (charging_access, anx_commute, income_per_car, home_sub excluded).
- Ensemble weights fixed a priori [0.40, 0.15, 0.15, 0.15, 0.15]; OOF is
  confirmatory, not hill-climbed (contrast v56/v37b weight selection on OOF).

## Leak risks
- All features fit-free (digits, ratios, uniform thresholds) except native
  categorical handling inside each fold's training data. No target encodings,
  no external rows, no test-label use.
- Band-swap / JZ-anchor lineage deliberately excluded: JZ has no train-fold
  predictions, so no JZ blend can carry complete comparable OOF.
- Honest caveat: lgb6/xgb5/memory weights and component choice reuse public
  knowledge; OOF CIs do not adjust for selection.

## Validation (per candidate)
Header id,Will_Buy_EV; 286571 rows; ids equal+ordered vs sample; finite; in
[0,1]. (Builder asserts all; hashes in manifest_muse.json.)

## Correlations / diversity
TBD after build (test-space Spearman vs memory/lgb6/xgb5/v57/v68).

## LB context (read-only, Sep 20)
Top 0.94675 Thimsen; Deotte 0.94672; yalpha 0.94667; BOB 0.94666.
Our best 0.94656 (v57-v59). Today: v71 0.94646, v72 0.94650 (both below best).
Verified-ensemble analog v68 (OOF 0.94627) scored 0.94645 public: OOF gains in
this family have not transferred 1:1 to public. Expected LB ranges below are
honest intervals, not promises; first place is not guaranteed.

## exp_log rows added
TBD (append-only, no prior-row edits).
