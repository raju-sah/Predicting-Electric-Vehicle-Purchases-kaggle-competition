import csv, os

rows = [
    ("v40", "JZ-0.94651-base + 10% hillclimbed stack + shifts + lexsort", "", "0.94650",
     "injection costs -0.00001 on the stronger base (it helped the old 0.94649 anchor)"),
    ("v41", "JZ-0.94651-base + 12% hillclimbed stack + shifts + lexsort", "", "0.94650",
     "A/B vs v38: same recipe on better base = still 0.94650; injection is the culprit"),
    ("v42", "JZ-0.94651-base + 15% hillclimbed stack + shifts + lexsort", "", "0.94650",
     "whole curve w=0.10/0.12/0.15 flat 0.94650 -> stack injection dead on JZ base"),
    ("v43", "JZ exact 0.94651 base raw (sha-verified, catalog own_001)", "0.94651", "0.94651",
     "NEW BEST. edge survives resubmission; secured the file + 2 sibling 0.94651 files (rank-identical trio)"),
    ("v44", "JZ base + raw boundary shifts 1x", "", "0.94651",
     "shifts EXACTLY free on this base (3731 rows moved, no score change)"),
    ("v45", "JZ base lexrank reorder only (nn secondary)", "", "0.94651",
     "reorder exactly free; -0.00001 decomposition: stack injection is the only harmful transform"),
    ("v46", "JZ base + 2x boundary shifts", "", "0.94651",
     "2x also exactly free -> shifts have no LB effect either way; base family is a local optimum"),
]

new = not os.path.exists("exp_log.csv")
with open("exp_log.csv", "a", newline="") as f:
    w = csv.writer(f)
    if new:
        w.writerow(["ver", "model", "OOF_AUC", "public_LB", "note"])
    for r in rows:
        w.writerow(r)
print("appended", len(rows), "rows")
