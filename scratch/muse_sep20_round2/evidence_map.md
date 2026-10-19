# Round-2 evidence map — 2026-09-20 (live)

Board: Thimsen 0.94675 (Sep 17), Deotte 0.94672, Elefante 0.94667 (NEW Sep 20),
yalpha 0.94667 (Sep 20), BOB 0.94666 (Sep 20). Our best 0.94656. Need +0.00019.

## Sources searched (Sep-20 fresh)

1. Competition notebooks by score: top is still the 0.94656 band-swap family
   (megayak reading-the-public-split, amanatar champion). No public notebook
   claims >0.94656. Top-team jumps (Elefante/yalpha/BOB) have no public artifacts.
2. Notebooks by dateRun (Sep 20): embedding-MLP (NN standalone ~0.9385, weak),
   pseudo15 (honest gated student, outputs cleared, folds unknown, no downloads),
   devashish (converges 0.9333–0.9419, below frontier), mahabub stack 0.94456,
   DefiAudit reverse-probes (negative result: every rule real, none help alone).
3. Datasets: defiaudit/s6e9-realmlp-oof (updated Sep 20 12:32) — DOWNLOADED, but
   contains test-only public probes, NO train OOF. Intel only: 50/50 and 75/25
   mega-champ rank blends score 0.94657 public (+0.00001 over 0.94656).
   jazivxt/s6e9-zoom-zoom-baseline refresh (Sep 19, +optimizer replay assets,
   history_catalog, latest_scored_results) — DOWNLOADING (2.7 GB).
4. Forums API: 403 forbidden. Discussion page webfetch empty. No forum intel.
5. Local: no concurrent-worker v71/v72 OOFs exist (their training never ran).
   v75–v78 family frozen per verdict (no submit/relabel).

## Routes evaluated

- R1 mega-champ 50/50 blend (0.94657 public): public-tuned, no train OOF
  possible → cannot be GO. Record as LB context only.
- R2 pseudo15-style gated rank-student: replicable locally on gate folds, new
  mechanism vs killed list (v10 linear stack overfit; this is a nonlinear
  shallow student, gated 4/5). Training decision AFTER disw result.
- R3 disagreement-weighted LGBM: TRAINED pooled 0.9435927 (folds
  0.94255/0.94326/0.94454/0.94390/0.94376); rho 0.9968 vs digit-solo proves the
  reweighting barely moves rankings; 6-way blends 0.94593 < v68. KILLED as
  candidate (logged r2_disw), kept as OOF donor.
- R4 embedding-MLP standalone: 0.9385 < gate → dead unless as micro-member.
- R5 JZ refresh optima: DOWNLOADED + audited. His five own 0.94656 files are
  pairwise rho≈0.999999 and rho≈0.999999 vs our v57 — same ranking family,
  micro-differences only. No new anchor (script compare_jz56.py, sha-verified).

## GO criteria (all required)

Complete seed-42 5-fold OOF >0.94230 AND reasoned public path >0.94656
(OOF near/above v68's 0.94627 with new diversity, or a new public-split
mechanism with OOF-neutral-or-better proof). Public-only scores never count.
