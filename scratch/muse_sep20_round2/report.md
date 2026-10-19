# S6E9 round-2 report — 2026-09-20/21: BLOCKER (no honest GO candidates)

Verdict context: round-1 v75–v78 rejected for submission strategy (solos NO-GO,
blends expected below 0.94656). This round searched exhaustively for a stronger
anchor, new public-split insight, or reproducible ranking transformation, then
tested the two best new mechanisms locally. Conclusion: four honest GO
candidates (complete OOF + plausible path above 0.94656) do not exist. Zero
candidates are shipped; nothing was submitted; no v79+ file was created; no
existing file was overwritten. exp_log.csv appended only (r2_disw, r2_student).

## 1. Sources searched (all Sep-20 fresh) and why each route fails

1. Competition notebooks by score: top remains the 0.94656 band-swap family.
   No public notebook claims above 0.94656. Top-team jumps (Elefante 0.94667,
   yalpha 0.94667, BOB 0.94666, all Sep 20) have zero public artifacts —
   nothing to reproduce or blend. Route fails: no downloadable stronger anchor.
2. Notebooks by dateRun: embedding-MLP (NN standalone ~0.9385 < gate 0.94230);
   devashish (converges 0.9333–0.9419, below frontier); mahabub stack 0.94456;
   DefiAudit reverse-probes (negative result by design). Route fails: all below
   frontier or negative.
3. defiaudit/s6e9-realmlp-oof (updated Sep 20): DOWNLOADED — contains only
   test-space public probes, no train OOF. Intel value only: author-measured
   50/50 and 75/25 mega-champ rank blends score 0.94657 public (+0.00001 over
   our 0.94656). Route fails as GO material: public-tuned by its author, no
   train-fold predictions exist, so no complete comparable OOF is possible.
   Reproducing it would be a public-only gamble, explicitly out of scope.
4. denpugovkin pseudo-labels-15-teachers (Sep 20): honest gated method, but
   stored outputs cleared, fold seed unrecoverable, no downloadable OOF, and
   its own submission.csv just reproduces 0.94656. Route fails as artifact;
   its MECHANISM was replicated locally instead (see §2).
5. jazivxt/s6e9-zoom-zoom-baseline refresh (Sep 19, 2.7 GB): DOWNLOADED,
   sha-verified. His five own 0.94656 files are pairwise rho≈0.999999 and
   rho≈0.999999 vs our v57 — the same ranking family, micro-differences only.
   Route fails: no new anchor, and JZ files are test-only (no complete OOF).
6. Forums API: 403 forbidden; discussion page fetch empty. No forum intel.
7. Local pool: concurrent worker's v71/v72 trainings never materialized OOFs;
   v75–v78 frozen per verdict. No hidden local asset.

## 2. New mechanisms trained locally (exact gate folds, both complete OOF)

- R3 disagreement-weighted LGBM (sample reweighting — never tried; weights
  from frozen OOF ranks only, k=3 clipped [1,5] a priori): pooled 0.9435927,
  gate-passing but rho 0.9968 vs digit-solo (same features) — the reweighting
  barely moves rankings; 6-way blends reach only 0.94593 < v68. KILLED as
  candidate, kept as OOF donor. (train_disw.py, oof_disw.npy, run_disw.json)
- R2 gated rank-student (pseudo15 mechanism on OUR folds: shallow LGBM on 6
  frozen OOF ranks + raw features, 20% dose into v78 teacher, acceptance bar
  pooled-delta>0 AND ≥4/5 fold wins): ACCEPTED the technical bar, +0.0000680
  with 5/5 fold wins, candidate pooled 0.9461334 (oof_stu_accepted_blend.npy,
  run_stu.json). BUT absolute level sits below v68 (0.9462709) and below the
  staged 4-way logit (0.9462450), and test-family rho is 0.9992 vs v68 — the
  same ranking family, no public-breakthrough path. NO-GO per strategy verdict.

## 3. Why no GO set can be honestly assembled

Every trainable route lands inside the 0.9460–0.9463 complete-OOF band whose
public precedent (v68: 0.94627 → 0.94645) points below 0.94656. Every route
above 0.94656 (mega-champ 50/50 at 0.94657, JZ's five 0.94656 files, nested
band probes) is public-tuned with no train-fold counterpart, hence
incompatible with the complete-OOF requirement by construction. The +0.00019
to #1 (0.94675) needs a better base ranking or new public-split insight;
neither exists in any searched source. Padding the set with the student blend
or disw variants would repeat exactly what the verdict rejected. So: zero
candidates shipped, by design.

## 4. Slot order / recommendation

No submissions recommended. If the coordinator wants a principled probe
despite this report, the least-bad documented option remains staged logit4
(0.9462450 complete OOF, manifest in muse_sep20/staging/) — still expected
below 0.94656. Do not manufacture v79+ GO labels for it.

## 5. Reproducibility (all new, scratch/muse_sep20_round2/)

evidence_map.md, scan_notebooks.py, read_pseudo15.py, read_pseudo15b.py,
scan_devashish.py, train_disw.py, check_disw.py, train_stu.py,
check_stu_corr.py, log_disw_row.py, log_stu_row.py, find_jz56.py,
compare_jz56.py, logs, run_disw.json, run_stu.json, oof_disw/test_disw,
oof_stu/test_stu/oof_stu_accepted_blend, nb_embedmlp/nb_pseudo15/nb_devashish,
defiaudit_realmlp/, jz_refresh/. Runtimes: disw ~11 min, student ~4 min.
