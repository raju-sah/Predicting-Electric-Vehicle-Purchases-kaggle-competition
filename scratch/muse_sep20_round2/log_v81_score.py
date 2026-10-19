"""Log v81 score (append-only)."""
import csv

with open("exp_log.csv", "a", newline="") as f:
    csv.writer(f).writerows([["v81_scored", "v81 public LB result", "", "0.94657",
                              "NEW BEST (ties author claim exactly); faithful recipe with "
                              "megayak own submission.csv; public-probe, no OOF; "
                              "need +0.00018 for #1 (0.94675)"]])
print("logged")
