with open("exp_log.csv", "a") as f:
    f.write("v30_jazivxt_sep14_pass2_baseline,Jazivxt Sep 14 Pass-2 Baseline (80% 0.94649 ref + 20% pass-2 LGBM),0.94612,0.94647,Second-pass LightGBM at 20% drags 0.94649 anchor down to 0.94647; confirms single model dilution\n")
print("Appended v30 to exp_log.csv")
