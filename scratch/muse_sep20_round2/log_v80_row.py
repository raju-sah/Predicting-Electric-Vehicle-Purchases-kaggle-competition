"""Log v80 outcome (append-only)."""
import csv

with open("exp_log.csv", "a", newline="") as f:
    csv.writer(f).writerows([
        ["v80", "Mega-champ 50/50 rank-blend reproduction (DefiAudit r8_m50)", "",
         "", "Public-probe reproduction per user order; NO complete OOF claimed; "
         "rho 0.99886 vs v57; SUBMITTED"],
        ["v80_scored", "v80 public LB result", "", "0.94647",
         "Claimed 0.94657 did NOT reproduce with our components "
         "(ds_megayak6 ensemble + r6 champ): -0.00010 gap; "
         "likely their mega file differs from our ensemble column; "
         "do not chase without understanding"]])
print("logged")
