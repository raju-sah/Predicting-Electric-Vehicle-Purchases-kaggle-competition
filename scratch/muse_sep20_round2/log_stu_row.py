"""Round2: log student outcome (append-only)."""
import csv

with open("exp_log.csv", "a", newline="") as f:
    csv.writer(f).writerows([[
        "r2_student", "Gated rank-student (pseudo15 mechanism, gate folds)", "0.9461334", "",
        "Shallow LGBM (7 leaves, min4000) on 6 frozen OOF ranks + raw features; "
        "20% dose into v78 teacher; ACCEPTED technical bar (+0.0000680, 5/5 fold wins); "
        "absolute 0.9461334 < v68 0.9462709 and < staged logit4 0.9462450; "
        "test-family rho 0.9992 vs v68 -> no public-breakthrough path; "
        "NO-GO per strategy verdict; NOT submitted; scratch/muse_sep20_round2/train_stu.py"]])
print("logged")
