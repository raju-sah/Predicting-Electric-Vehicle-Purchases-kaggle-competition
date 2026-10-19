with open("exp_log.csv", "a") as f:
    f.write("v34_apex_power_residual_ensemble,Apex Power-Residual Tiebreaker (92% Master 49/50 Anchor + 8% (Stacker+Deotte+RealMLP)^0.74 + Lexsort),0.94650,0.94648,Micro-perturbations at 8% non-linear residual slightly drop by 0.00001; confirms strict anchor preservation envelope\n")
print("Appended v34 to exp_log.csv")
