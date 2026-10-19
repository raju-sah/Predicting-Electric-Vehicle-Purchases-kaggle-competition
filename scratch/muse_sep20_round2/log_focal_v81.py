"""Log focal + v80/v81 rows (append-only)."""
import csv

with open("exp_log.csv", "a", newline="") as f:
    csv.writer(f).writerows([
        ["r2_focal", "Focal-loss LGBM gamma=2 alpha=0.25 (a priori)", "0.9436045", "",
         "NEW local mechanism (no focal ever trained here); digit frame; gate folds; "
         "~0.9436 ~= disw, below digit 0.94379; no gain, no-go; NOT submitted"],
        ["v81", "True mega56-champ 50/50 (megayak own submission.csv)", "", "",
         "Faithful r8_m50 recipe: author's mega rank-matches v57 (rho 1.0); v80 used "
         "wrong ensemble column; public-probe, NO OOF; SUBMITTED per user order"]])
print("logged")
