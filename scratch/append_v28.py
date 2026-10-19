with open("exp_log.csv", "a") as f:
    f.write("v28_dual_anchor_lexsort_blend,Dual-Anchor Grandmaster (85% v26 + 10% v18 + 3% Stacker + 2% Deotte + Lexsort),0.94650,0.94649,Maintains 0.94649 Rank 20; high structural stability across two independent anchors\n")
print("Appended v28 to exp_log.csv")
