"""Round2: log disw outcome to exp_log (append-only)."""
import csv

with open("exp_log.csv", "a", newline="") as f:
    csv.writer(f).writerows([[
        "r2_disw", "Disagreement-weighted LGBM (k=3 clipped [1,5])", "0.9435927", "",
        "NEW mechanism (sample reweighting); weights from frozen OOF ranks only; "
        "seed-42 5-fold; folds 0.94255/0.94326/0.94454/0.94390/0.94376; "
        "rho 0.9968 vs digit (same features) -> ranking novelty nil; "
        "6-way blends 0.94593 below v68; KILLED as candidate, kept as OOF donor; "
        "NOT submitted; scratch/muse_sep20_round2/train_disw.py"]])
print("logged")
