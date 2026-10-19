"""Log MLP outcome (append-only)."""
import csv

with open("exp_log.csv", "a", newline="") as f:
    csv.writer(f).writerows([["r2_mlp", "sklearn MLP(128,64) scaled digit frame", "0.9383752", "",
                              "NEW family locally (neural); folds "
                              "0.93704/0.93863/0.93939/0.93905/0.93810; BELOW gate 0.94230; "
                              "KILLED as candidate and as donor (too weak); GPU RealMLP "
                              "escalation not justified on this signal; NOT submitted"]])
print("logged")
